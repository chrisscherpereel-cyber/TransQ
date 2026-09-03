"""Lecture Quiz Builder — Streamlit app.

Upload lecture audio -> local Whisper transcript -> LLM summary -> reviewable
multiple-choice bank -> exports for Canvas, Excel, Word, and PDF.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then point Streamlit Community Cloud at app.py
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile

import streamlit as st

from src import __version__
from src.chunking import allocate_questions
from src.config import AUDIO_EXTENSIONS, LLM_MODELS, WHISPER_MODELS, AppSettings, get_secret
from src.exporters import (
    export_csv,
    export_docx,
    export_markdown,
    export_pdf,
    export_qti12_canvas,
    export_qti21,
    export_xlsx,
)
from src.exporters.transcript_formats import (
    export_srt,
    export_timestamped_txt,
    export_txt,
    export_vtt,
)
from src.llm import LLMClient, LLMError, estimate_cost
from src.mcq import (
    balance_answer_positions,
    coverage_report,
    critique_and_revise,
    generate_questions,
    validate_all,
)
from src.schema import OPTION_LETTERS, Quiz, QuizMeta, Summary, format_timestamp
from src.summarize import summarize_transcript
from src.transcribe import (
    TranscriptionError,
    estimate_transcription_minutes,
    load_model,
    probe_duration,
    transcribe_file,
)

st.set_page_config(
    page_title="Lecture Quiz Builder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_whisper(model_size: str, compute_type: str):
    """Whisper weights are large; load them once per app process."""
    return load_model(model_size, compute_type)


def init_state() -> None:
    defaults = {
        "transcript": None,
        "summary": None,
        "quiz": None,
        "chunks": [],
        "usage_cost": 0.0,
        "source_filename": "",
        "authenticated": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Optional shared-password gate
# --------------------------------------------------------------------------- #


def password_gate() -> bool:
    expected = get_secret("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🎓 Lecture Quiz Builder")
    st.caption("This deployment is password protected.")
    with st.form("auth"):
        entered = st.text_input("Access password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if entered == expected:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def sidebar() -> AppSettings:
    s = AppSettings()
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        with st.expander("Transcription", expanded=True):
            s.whisper_model = st.selectbox(
                "Whisper model",
                list(WHISPER_MODELS),
                index=list(WHISPER_MODELS).index("small"),
                help="Larger models are more accurate and much slower.",
            )
            st.caption(WHISPER_MODELS[s.whisper_model])
            lang = st.selectbox(
                "Language", ["Auto-detect", "en", "es", "fr", "de", "zh", "hi", "pt"], index=0
            )
            s.language = None if lang == "Auto-detect" else lang
            s.vad_filter = st.checkbox(
                "Skip silence (voice-activity filter)",
                value=True,
                help="Speeds things up and prevents hallucinated text in quiet stretches. "
                "Turn off if quiet speech is being dropped.",
            )
            s.beam_size = st.slider("Beam size", 1, 5, 1, help="Higher is slower and slightly more accurate.")

        with st.expander("Question generation", expanded=True):
            s.provider = st.radio(
                "LLM provider", ["anthropic", "openai"], horizontal=True,
                format_func=lambda p: "Claude" if p == "anthropic" else "OpenAI",
            )
            s.llm_model = st.selectbox("Model", LLM_MODELS[s.provider])

            key_env = "ANTHROPIC_API_KEY" if s.provider == "anthropic" else "OPENAI_API_KEY"
            if get_secret(key_env):
                st.success(f"{key_env} found in secrets", icon="✅")
            else:
                s.api_key = st.text_input(
                    f"{key_env}", type="password",
                    help="Stored only for this browser session. For a shared deployment, "
                    "put it in Streamlit secrets instead.",
                )

            s.num_questions = st.slider("Number of questions", 3, 40, 10)
            s.options_per_question = st.select_slider("Options per question", [3, 4, 5], value=4)
            s.bloom_targets = st.multiselect(
                "Cognitive levels (Bloom)",
                ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
                default=["Remember", "Understand", "Apply", "Analyze"],
            )
            s.difficulty_mix = st.select_slider(
                "Difficulty", ["Mostly easy", "Balanced", "Mostly hard"], value="Balanced"
            )
            s.temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)

        with st.expander("Context & advanced"):
            s.course_context = st.text_area(
                "Course context (optional)",
                placeholder="MGT 301 Operations Management, junior-level. Emphasize "
                "capacity planning and the trade-offs between chase and level strategies.",
                height=110,
                help="Steers what the model treats as important. Does not add outside content.",
            )
            s.chunk_seconds = st.slider("Chunk length (minutes)", 3, 20, 10) * 60
            s.compute_type = st.selectbox(
                "Compute type", ["int8", "int8_float16", "float16", "float32"], index=0,
                help="int8 is the right choice on CPU. Use float16 only with a GPU.",
            )

        st.divider()
        if st.session_state.usage_cost:
            st.metric("Estimated API cost", f"${st.session_state.usage_cost:.4f}")
        st.caption(f"v{__version__} · faster-whisper runs locally; audio never leaves this server.")
    return s


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #


def run_transcription(uploaded, settings: AppSettings) -> None:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        audio_path = tmp.name

    try:
        duration = probe_duration(audio_path)
        if duration:
            est = estimate_transcription_minutes(duration, settings.whisper_model)
            st.info(
                f"Audio length {format_timestamp(duration)} · "
                f"estimated transcription time ~{est:.1f} min on this server."
            )

        bar = st.progress(0.0, text="Loading the Whisper model…")
        model = get_whisper(settings.whisper_model, settings.compute_type)

        def on_progress(frac: float, message: str) -> None:
            bar.progress(min(1.0, frac), text=message)

        transcript = transcribe_file(
            model,
            audio_path,
            language=settings.language,
            vad_filter=settings.vad_filter,
            beam_size=settings.beam_size,
            progress=on_progress,
        )
        bar.empty()

        st.session_state.transcript = transcript
        st.session_state.summary = None
        st.session_state.quiz = None
        st.session_state.source_filename = uploaded.name
        st.success(
            f"Transcribed {format_timestamp(transcript.duration)} of audio — "
            f"{transcript.word_count:,} words, {len(transcript.segments)} segments."
        )
    except TranscriptionError as exc:
        st.error(str(exc))
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def build_client(settings: AppSettings) -> LLMClient | None:
    try:
        return LLMClient(
            provider=settings.provider,
            model=settings.llm_model,
            api_key=settings.resolved_api_key(),
            temperature=settings.temperature,
        )
    except LLMError as exc:
        st.error(str(exc))
        return None


def run_generation(settings: AppSettings, do_review: bool) -> None:
    transcript = st.session_state.transcript
    client = build_client(settings)
    if client is None or transcript is None:
        return

    bar = st.progress(0.0, text="Summarizing…")

    try:
        summary, chunks, _ = summarize_transcript(
            client,
            transcript,
            course_context=settings.course_context,
            chunk_seconds=settings.chunk_seconds,
            overlap_seconds=settings.chunk_overlap_seconds,
            progress=lambda f, m: bar.progress(f * 0.4, text=m),
        )
        st.session_state.summary = summary
        st.session_state.chunks = chunks

        allocation = allocate_questions(settings.num_questions, len(chunks))
        questions = generate_questions(
            client,
            chunks,
            allocation,
            n_options=settings.options_per_question,
            bloom_targets=settings.bloom_targets,
            difficulty_mix=settings.difficulty_mix,
            course_context=settings.course_context,
            progress=lambda f, m: bar.progress(0.4 + f * 0.4, text=m),
        )

        if do_review and questions:
            questions, _ = critique_and_revise(
                client, questions, progress=lambda f, m: bar.progress(0.8 + f * 0.15, text=m)
            )

        questions = balance_answer_positions(questions)
        questions = validate_all(questions)

        st.session_state.quiz = Quiz(
            meta=QuizMeta(
                title=summary.title or "Lecture Quiz",
                course=settings.course_context.splitlines()[0][:80]
                if settings.course_context
                else "",
                description=summary.abstract[:400],
                source_filename=st.session_state.source_filename,
                generated_on=dt.date.today().isoformat(),
                model_used=f"{settings.llm_model} + whisper-{settings.whisper_model}",
            ),
            questions=questions,
        )
        st.session_state.usage_cost = estimate_cost(settings.llm_model, client.usage)
        bar.progress(1.0, text="Done")
        bar.empty()

        if not questions:
            st.warning("No questions could be generated. Try a different model or a longer clip.")
        else:
            st.success(f"Generated {len(questions)} questions.")
    except LLMError as exc:
        bar.empty()
        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def transcript_tab() -> None:
    transcript = st.session_state.transcript
    if transcript is None:
        st.info("Upload an audio file to get started.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", format_timestamp(transcript.duration))
    c2.metric("Words", f"{transcript.word_count:,}")
    c3.metric("Segments", len(transcript.segments))
    c4.metric("Language", transcript.language.upper())

    show_times = st.toggle("Show timestamps", value=True)
    text = transcript.text_with_timestamps() if show_times else transcript.text
    st.text_area("Transcript", text, height=440, label_visibility="collapsed")

    st.markdown("**Download transcript**")
    d1, d2, d3, d4 = st.columns(4)
    stem = os.path.splitext(st.session_state.source_filename or "lecture")[0]
    d1.download_button("Plain text", export_txt(transcript), f"{stem}.txt", "text/plain",
                       use_container_width=True)
    d2.download_button("Timestamped", export_timestamped_txt(transcript),
                       f"{stem}_timestamped.txt", "text/plain", use_container_width=True)
    d3.download_button("SRT captions", export_srt(transcript), f"{stem}.srt", "text/plain",
                       use_container_width=True)
    d4.download_button("WebVTT", export_vtt(transcript), f"{stem}.vtt", "text/vtt",
                       use_container_width=True)


def summary_tab() -> None:
    summary: Summary | None = st.session_state.summary
    if summary is None:
        st.info("Run **Summarize & generate questions** to see the summary.")
        return

    if summary.title:
        st.subheader(summary.title)
    if summary.abstract:
        st.write(summary.abstract)

    left, right = st.columns(2)
    with left:
        if summary.learning_objectives:
            st.markdown("#### Learning objectives")
            for obj in summary.learning_objectives:
                st.markdown(f"- {obj}")
        if summary.key_points:
            st.markdown("#### Key points")
            for point in summary.key_points:
                st.markdown(f"- {point}")
    with right:
        if summary.outline:
            st.markdown("#### Outline")
            for item in summary.outline:
                st.markdown(
                    f"**{item.get('timestamp', '')} · {item.get('heading', '')}**  \n"
                    f"{item.get('detail', '')}"
                )
        if summary.key_terms:
            st.markdown("#### Key terms")
            for term in summary.key_terms:
                st.markdown(f"- **{term.get('term', '')}** — {term.get('definition', '')}")

    st.download_button(
        "Download summary (Markdown)",
        summary.as_markdown().encode("utf-8"),
        "lecture_summary.md",
        "text/markdown",
    )


def questions_tab() -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None or not quiz.questions:
        st.info("Run **Summarize & generate questions** to build the question bank.")
        return

    cov = coverage_report(quiz.included)
    flagged = sum(1 for q in quiz.questions if q.flags)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Included", f"{len(quiz.included)} / {len(quiz.questions)}")
    c2.metric("Points", f"{quiz.total_points:g}")
    c3.metric("Flagged for review", flagged)
    spread = cov["answer_position"]
    c4.metric("Answer spread", " ".join(f"{k}:{v}" for k, v in sorted(spread.items())))

    if flagged:
        st.warning(
            f"{flagged} question(s) tripped an item-writing check. Expand them below — "
            "the flags are advisory, not automatic rejections.",
            icon="⚠️",
        )

    with st.expander("Coverage breakdown"):
        b1, b2 = st.columns(2)
        b1.markdown("**Bloom level**")
        b1.bar_chart(cov["bloom"])
        b2.markdown("**Difficulty**")
        b2.bar_chart(cov["difficulty"])

    st.divider()

    for i, q in enumerate(quiz.questions):
        badge = "⚠️ " if q.flags else ""
        state = "" if q.include else " · excluded"
        with st.expander(f"{badge}Q{i + 1}. {q.stem[:90]}{'…' if len(q.stem) > 90 else ''}{state}"):
            top = st.columns([1, 1, 1, 1])
            q.include = top[0].checkbox("Include", value=q.include, key=f"inc_{q.id}")
            top[1].caption(f"**{q.bloom}** · {q.difficulty}")
            top[2].caption(f"⏱ {q.source_timestamp}")
            q.points = top[3].number_input(
                "Points", 0.0, 100.0, float(q.points), 0.5, key=f"pts_{q.id}"
            )

            q.stem = st.text_area("Question", q.stem, key=f"stem_{q.id}", height=80)

            new_options: list[str] = []
            for j, opt in enumerate(q.options):
                new_options.append(
                    st.text_input(
                        f"Option {OPTION_LETTERS[j]}"
                        + ("  ✅ correct" if j == q.correct_index else ""),
                        opt,
                        key=f"opt_{q.id}_{j}",
                    )
                )
            q.options = new_options
            q.correct_index = st.radio(
                "Correct answer",
                range(len(q.options)),
                index=q.correct_index,
                horizontal=True,
                format_func=lambda j: OPTION_LETTERS[j],
                key=f"corr_{q.id}",
            )

            if q.rationale:
                st.caption(f"**Why:** {q.rationale}")
            if q.source_quote:
                st.caption(f"**From the lecture:** “{q.source_quote}”")
            if q.flags:
                for flag in q.flags:
                    st.caption(f"⚠️ {flag}")


def export_tab() -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None or not quiz.included:
        st.info("Generate and select at least one question to enable exports.")
        return

    quiz.meta.title = st.text_input("Quiz title", quiz.meta.title)
    quiz.meta.course = st.text_input("Course label (optional)", quiz.meta.course)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in quiz.meta.title)[:60] or "quiz"
    summary = st.session_state.summary

    st.markdown("#### Learning management system")
    l1, l2 = st.columns(2)
    l1.download_button(
        "⬇️ QTI 1.2 · Canvas (.zip)",
        export_qti12_canvas(quiz),
        f"{stem}_qti12.zip",
        "application/zip",
        use_container_width=True,
        help="Canvas → Settings → Import Course Content → QTI .zip file. "
        "Also accepted by Blackboard, D2L, and Moodle.",
    )
    l2.download_button(
        "⬇️ QTI 2.1 · IMS standard (.zip)",
        export_qti21(quiz),
        f"{stem}_qti21.zip",
        "application/zip",
        use_container_width=True,
        help="Use if your LMS specifically requires QTI 2.1.",
    )

    st.markdown("#### Spreadsheet")
    s1, s2 = st.columns(2)
    s1.download_button("⬇️ Excel workbook (.xlsx)", export_xlsx(quiz), f"{stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    s2.download_button("⬇️ CSV", export_csv(quiz), f"{stem}.csv", "text/csv",
                       use_container_width=True)

    st.markdown("#### Printable")
    include_key = st.checkbox("Include instructor answer key", value=True)
    include_summary = st.checkbox("Include lecture summary in the document", value=False)
    doc_summary = summary if include_summary else None

    p1, p2, p3 = st.columns(3)
    p1.download_button(
        "⬇️ Word (.docx)",
        export_docx(quiz, doc_summary, include_key),
        f"{stem}.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    p2.download_button("⬇️ PDF", export_pdf(quiz, doc_summary, include_key), f"{stem}.pdf",
                       "application/pdf", use_container_width=True)
    p3.download_button("⬇️ Markdown", export_markdown(quiz, doc_summary), f"{stem}.md",
                       "text/markdown", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    init_state()
    if not password_gate():
        return

    settings = sidebar()

    st.title("🎓 Lecture Quiz Builder")
    st.caption(
        "Transcribe a lecture with Whisper, summarize it, and generate a reviewable "
        "multiple-choice bank you can import straight into your LMS."
    )

    uploaded = st.file_uploader(
        "Lecture audio or video",
        type=AUDIO_EXTENSIONS,
        help="Up to ~400 MB. Longer recordings take proportionally longer to transcribe.",
    )

    a1, a2, _ = st.columns([1, 1, 2])
    if a1.button(
        "1 · Transcribe", type="primary", disabled=uploaded is None, use_container_width=True
    ):
        run_transcription(uploaded, settings)

    review = st.sidebar.checkbox(
        "Run a second-pass quality review",
        value=True,
        help="A reviewer prompt re-reads the drafted items and repairs or drops weak ones. "
        "Roughly doubles generation cost and is the single biggest quality gain here.",
    )
    if a2.button(
        "2 · Summarize & generate",
        disabled=st.session_state.transcript is None,
        use_container_width=True,
    ):
        run_generation(settings, review)

    st.divider()
    t1, t2, t3, t4 = st.tabs(["📝 Transcript", "📋 Summary", "❓ Questions", "⬇️ Export"])
    with t1:
        transcript_tab()
    with t2:
        summary_tab()
    with t3:
        questions_tab()
    with t4:
        export_tab()


if __name__ == "__main__":
    main()
