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
from src.config import (
    AUDIO_EXTENSIONS,
    DEFAULT_PROVIDER,
    PROVIDERS,
    WHISPER_MODELS,
    AppSettings,
    get_provider,
    get_secret,
)
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
from src.llm import LLMClient, LLMError, estimate_cost, has_pricing, register_pricing
from src.openrouter_catalog import ORModel, load_models, pricing_map, vendors
from src.mcq import (
    balance_answer_positions,
    coverage_report,
    critique_and_revise,
    generate_questions,
    generate_replacement,
    validate_all,
)
from src.schema import OPTION_LETTERS, Quiz, QuizMeta, Summary, Transcript, format_timestamp
from src.summarize import summarize_transcript
from src.transcribe import (
    TranscriptionError,
    estimate_transcription_minutes,
    load_model,
    natural_sort_key,
    probe_duration,
    transcribe_parts,
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


@st.cache_data(ttl=3600, show_spinner="Loading the OpenRouter model list…")
def get_openrouter_models(_nonce: int = 0) -> tuple[list[ORModel], str | None]:
    """The live catalog, refreshed hourly.

    ``_nonce`` is not used by the function — bumping it is how the refresh button
    busts Streamlit's cache without waiting out the hour.
    """
    return load_models()


def init_state() -> None:
    defaults = {
        "transcript": None,
        "summary": None,
        "quiz": None,
        "quiz_versions": [],
        "active_version": 0,
        "chunks": [],
        "usage_cost": 0.0,
        "usage_tokens": 0,
        "usage_priced": True,
        "source_filename": "",
        "authenticated": False,
        "catalog_nonce": 0,
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


def openrouter_model_picker(default_slug: str) -> str:
    """Every model OpenRouter currently carries, A–Z, free ones marked.

    The list is fetched live rather than hardcoded: OpenRouter's roster turns
    over weekly, so a baked-in list would offer retired models and hide new ones.
    """
    catalog, warning = get_openrouter_models(st.session_state.catalog_nonce)
    register_pricing(pricing_map(catalog))

    if warning:
        st.warning(warning, icon="📶")

    free_only = st.checkbox(
        "Free models only",
        value=False,
        help="Models OpenRouter serves at $0. They are rate-limited and often "
        "smaller — fine for trying the app out, weaker at following the "
        "item-writing rules.",
    )
    vendor_names = vendors(catalog)
    chosen_vendors = st.multiselect(
        "Filter by vendor", vendor_names, default=[], placeholder="All vendors"
    )

    shown = [
        m
        for m in catalog
        if (not free_only or m.is_free)
        and (not chosen_vendors or m.vendor in chosen_vendors)
    ]

    if not shown:
        st.info("No models match those filters.")
        shown = catalog

    slugs = [m.id for m in shown]
    labels = {m.id: m.option_label for m in shown}
    index = slugs.index(default_slug) if default_slug in slugs else 0

    free_count = sum(1 for m in catalog if m.is_free)
    st.caption(
        f"{len(shown)} of {len(catalog)} models · {free_count} free · "
        f"{'bundled snapshot' if warning else 'live from openrouter.ai'}"
    )

    selected = st.selectbox(
        "Model",
        slugs,
        index=index,
        format_func=lambda slug: labels.get(slug, slug),
        help="Type to search. Sorted alphabetically; 🆓 marks models priced at $0.",
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("↻ Refresh list", use_container_width=True):
        st.session_state.catalog_nonce += 1
        get_openrouter_models.clear()
        st.rerun()
    custom = c2.text_input(
        "Or a slug", value="", placeholder="vendor/model",
        help="Anything not in the list — a brand-new model, or a variant.",
    ).strip()

    return custom or selected


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
            provider_keys = list(PROVIDERS)
            s.provider = st.selectbox(
                "LLM provider",
                provider_keys,
                index=provider_keys.index(DEFAULT_PROVIDER),
                format_func=lambda k: PROVIDERS[k].label,
            )
            spec = get_provider(s.provider)
            if spec.note:
                st.caption(spec.note)

            if s.provider == "openrouter":
                s.llm_model = openrouter_model_picker(spec.models[0])
            else:
                models = list(spec.models)
                s.llm_model = st.selectbox("Model", models, key=f"model_{s.provider}")

            if get_secret(spec.env_var):
                st.success(f"{spec.env_var} found in secrets", icon="✅")
            else:
                s.api_key = st.text_input(
                    spec.env_var,
                    type="password",
                    help="Stored only for this browser session. For a shared deployment, "
                    "put it in Streamlit secrets instead.",
                )
                if spec.console_url:
                    st.caption(f"[Get a {spec.label} key]({spec.console_url})")

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
        if st.session_state.usage_tokens:
            c1, c2 = st.columns(2)
            c1.metric("Tokens used", f"{st.session_state.usage_tokens:,}")
            if st.session_state.usage_priced:
                c2.metric(
                    "Est. API cost",
                    f"${st.session_state.usage_cost:.4f}",
                    help="Approximate. OpenRouter routes to whichever upstream host is "
                    "cheapest at the moment, so check its dashboard for actual spend."
                    if s.provider == "openrouter"
                    else "Based on list prices at the time of writing.",
                )
            else:
                c2.metric("Est. API cost", "—", help="No list price on file for this model.")
        st.caption(f"v{__version__} · faster-whisper runs locally; audio never leaves this server.")
    return s


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #


def order_uploads(files: list, use_upload_order: bool) -> list:
    """Put the parts of a split recording into lecture order."""
    if use_upload_order:
        return list(files)
    return sorted(files, key=lambda f: natural_sort_key(f.name))


def run_transcription(uploaded: list, settings: AppSettings) -> None:
    """Transcribe one file, or several parts of a split recording, in order."""
    paths: list[str] = []
    names: list[str] = []

    for item in uploaded:
        suffix = os.path.splitext(item.name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(item.getbuffer())
            paths.append(tmp.name)
            names.append(item.name)

    try:
        durations = [probe_duration(p) for p in paths]
        total = sum(durations)
        if total:
            est = estimate_transcription_minutes(total, settings.whisper_model)
            noun = "part" if len(paths) == 1 else "parts"
            st.info(
                f"{len(paths)} {noun} · combined length {format_timestamp(total)} · "
                f"estimated transcription time ~{est:.1f} min on this server."
            )

        bar = st.progress(0.0, text="Loading the Whisper model…")
        model = get_whisper(settings.whisper_model, settings.compute_type)

        transcript = transcribe_parts(
            model,
            paths,
            display_names=names,
            language=settings.language,
            vad_filter=settings.vad_filter,
            beam_size=settings.beam_size,
            progress=lambda f, m: bar.progress(min(1.0, f), text=m),
        )
        bar.empty()

        st.session_state.transcript = transcript
        st.session_state.summary = None
        st.session_state.quiz = None
        st.session_state.quiz_versions = []
        st.session_state.active_version = 0
        st.session_state.source_filename = (
            names[0] if len(names) == 1 else f"{os.path.splitext(names[0])[0]}_combined"
        )

        st.success(
            f"Transcribed {format_timestamp(transcript.duration)} of audio across "
            f"{len(transcript.parts)} part(s) — {transcript.word_count:,} words, "
            f"{len(transcript.segments)} segments."
        )
        if transcript.skipped_parts:
            st.warning(
                "Some parts were skipped and are missing from the transcript:\n\n"
                + "\n".join(f"- {s}" for s in transcript.skipped_parts),
                icon="⚠️",
            )
    except TranscriptionError as exc:
        st.error(str(exc))
    finally:
        for path in paths:
            try:
                os.unlink(path)
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


def record_usage(client: LLMClient, settings: AppSettings) -> None:
    st.session_state.usage_cost += estimate_cost(settings.llm_model, client.usage)
    st.session_state.usage_tokens += client.usage.input_tokens + client.usage.output_tokens
    st.session_state.usage_priced = has_pricing(settings.llm_model)


def all_existing_stems() -> list[str]:
    """Every stem the instructor has already seen, across all versions."""
    stems: list[str] = []
    for quiz in st.session_state.quiz_versions:
        stems.extend(q.stem for q in quiz.questions)
    return stems


def store_version(quiz: Quiz) -> None:
    quiz.meta.title = quiz.meta.title or "Lecture Quiz"
    st.session_state.quiz_versions.append(quiz)
    st.session_state.active_version = len(st.session_state.quiz_versions) - 1
    st.session_state.quiz = quiz


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

        store_version(build_quiz(questions, summary, settings, version=1))
        record_usage(client, settings)
        bar.progress(1.0, text="Done")
        bar.empty()

        if not questions:
            st.warning("No questions could be generated. Try a different model or a longer clip.")
        else:
            st.success(f"Generated {len(questions)} questions.")
    except LLMError as exc:
        bar.empty()
        st.error(str(exc))


def build_quiz(questions, summary: Summary, settings: AppSettings, version: int) -> Quiz:
    base = summary.title or "Lecture Quiz"
    return Quiz(
        meta=QuizMeta(
            title=base if version == 1 else f"{base} — Set {version}",
            course=settings.course_context.splitlines()[0][:80]
            if settings.course_context
            else "",
            description=summary.abstract[:400],
            source_filename=st.session_state.source_filename,
            generated_on=dt.date.today().isoformat(),
            model_used=f"{get_provider(settings.provider).label} {settings.llm_model} "
            f"+ whisper-{settings.whisper_model}",
        ),
        questions=questions,
    )


def run_alternative_set(settings: AppSettings, do_review: bool) -> bool:
    """Generate a fresh set of questions over the same lecture.

    The existing questions are passed in as an explicit avoid-list. Without
    that, a second run over the same transcript reproduces the first one almost
    verbatim — the model keeps finding the same salient points, because they are
    the same salient points.
    """
    chunks = st.session_state.chunks
    summary = st.session_state.summary
    client = build_client(settings)
    if client is None or not chunks or summary is None:
        st.error("Run the first generation pass before asking for an alternative set.")
        return False

    bar = st.progress(0.0, text="Writing an alternative set…")
    try:
        allocation = allocate_questions(settings.num_questions, len(chunks))
        questions = generate_questions(
            client,
            chunks,
            allocation,
            n_options=settings.options_per_question,
            bloom_targets=settings.bloom_targets,
            difficulty_mix=settings.difficulty_mix,
            course_context=settings.course_context,
            avoid_stems=all_existing_stems(),
            progress=lambda f, m: bar.progress(f * 0.8, text=m),
        )
        if do_review and questions:
            questions, _ = critique_and_revise(
                client, questions, progress=lambda f, m: bar.progress(0.8 + f * 0.15, text=m)
            )
        questions = validate_all(balance_answer_positions(questions))

        bar.empty()
        if not questions:
            st.warning(
                "The alternative pass produced nothing usable. "
                "Try again, or raise the temperature in the sidebar."
            )
            return False

        store_version(
            build_quiz(
                questions, summary, settings, version=len(st.session_state.quiz_versions) + 1
            )
        )
        record_usage(client, settings)
        # A toast rather than st.success: the caller reruns to show the new set,
        # and a success box would be wiped by that rerun before it was read.
        st.toast(
            f"Added Set {len(st.session_state.quiz_versions)} — {len(questions)} new questions.",
            icon="✨",
        )
        return True
    except LLMError as exc:
        bar.empty()
        st.error(str(exc))
        return False


def run_replacement(settings: AppSettings, index: int, same_section: bool = True) -> None:
    """Swap one question for a newly written one."""
    quiz: Quiz | None = st.session_state.quiz
    chunks = st.session_state.chunks
    if quiz is None or not chunks:
        return
    client = build_client(settings)
    if client is None:
        return

    old = quiz.questions[index]
    try:
        with st.spinner("Writing a replacement question…"):
            new = generate_replacement(
                client,
                chunks,
                old,
                existing_stems=all_existing_stems(),
                n_options=settings.options_per_question,
                bloom_targets=settings.bloom_targets,
                difficulty_mix=settings.difficulty_mix,
                course_context=settings.course_context,
                same_section=same_section,
            )
        record_usage(client, settings)
    except LLMError as exc:
        st.error(str(exc))
        return

    if new is None:
        st.warning("Could not write a replacement for that question. Try again.")
        return

    quiz.questions[index] = new
    validate_all(quiz.questions)
    st.toast(f"Replaced question {index + 1}.", icon="🔄")
    st.rerun()


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def transcript_tab() -> None:
    transcript: Transcript | None = st.session_state.transcript
    if transcript is None:
        st.info("Upload an audio file to get started.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", format_timestamp(transcript.duration))
    c2.metric("Words", f"{transcript.word_count:,}")
    c3.metric("Segments", len(transcript.segments))
    c4.metric("Parts" if transcript.is_multipart else "Language",
              len(transcript.parts) if transcript.is_multipart else transcript.language.upper())

    if transcript.is_multipart:
        with st.expander("Parts stitched into this transcript", expanded=False):
            st.caption(
                "Timestamps below are on the combined lecture timeline, so a question "
                "tagged 0:52:14 points at the same moment whether the recording arrived "
                "as one file or five."
            )
            for part in transcript.parts:
                st.markdown(f"- {part.label} — {part.segments} segments")

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


def version_bar(settings: AppSettings, review: bool) -> None:
    """Switch between generated sets, or ask for another one."""
    versions = st.session_state.quiz_versions
    left, right = st.columns([3, 2])

    with left:
        if len(versions) > 1:
            labels = [
                f"Set {i + 1} ({len(v.included)} questions)" for i, v in enumerate(versions)
            ]
            picked = st.radio(
                "Question set",
                range(len(versions)),
                index=st.session_state.active_version,
                format_func=lambda i: labels[i],
                horizontal=True,
            )
            if picked != st.session_state.active_version:
                st.session_state.active_version = picked
                st.session_state.quiz = versions[picked]
                st.rerun()
        else:
            st.caption("One set generated so far.")

    with right:
        if st.button(
            "✨ Generate an alternative set",
            use_container_width=True,
            help="Writes a whole new set over the same lecture, steered away from "
            "every question you already have. The old set is kept — switch between "
            "them on the left.",
        ):
            if run_alternative_set(settings, review):
                st.rerun()


def questions_tab(settings: AppSettings, review: bool) -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None or not quiz.questions:
        st.info("Run **Summarize & generate questions** to build the question bank.")
        return

    version_bar(settings, review)
    st.divider()

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

            st.divider()
            r1, r2 = st.columns([1, 2])
            same_section = r2.toggle(
                "Draw the replacement from the same part of the lecture",
                value=True,
                key=f"same_{q.id}",
                help="On: the new question comes from the same stretch of the "
                "recording, so coverage stays even. Off: anywhere in the lecture.",
            )
            if r1.button(
                "🔄 Replace this question",
                key=f"repl_{q.id}",
                use_container_width=True,
                disabled=not st.session_state.chunks,
            ):
                run_replacement(settings, i, same_section=same_section)


def export_tab() -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None or not quiz.included:
        st.info("Generate and select at least one question to enable exports.")
        return

    version = st.session_state.active_version
    if len(st.session_state.quiz_versions) > 1:
        st.caption(
            f"Exporting **Set {version + 1}** of "
            f"{len(st.session_state.quiz_versions)}. Switch sets on the Questions tab."
        )
    # Keys are version-scoped so switching sets does not carry the previous
    # set's title into the box.
    quiz.meta.title = st.text_input("Quiz title", quiz.meta.title, key=f"title_{version}")
    quiz.meta.course = st.text_input(
        "Course label (optional)", quiz.meta.course, key=f"course_{version}"
    )
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
        "Lecture audio or video — one file, or several parts of a split recording",
        type=AUDIO_EXTENSIONS,
        accept_multiple_files=True,
        help="Split a long lecture into parts if a single file is too big or too slow. "
        "The parts are transcribed in order and stitched into one continuous transcript.",
    )

    ordered: list = []
    if uploaded:
        use_upload_order = False
        if len(uploaded) > 1:
            use_upload_order = st.checkbox(
                "Use the order I uploaded them in",
                value=False,
                help="Off: parts are ordered by filename, with numbers read as numbers "
                "(so part2 comes before part10).",
            )
        ordered = order_uploads(uploaded, use_upload_order)
        if len(ordered) > 1:
            st.caption("**Transcription order:** " + " → ".join(f.name for f in ordered))

    a1, a2, _ = st.columns([1, 1, 2])
    if a1.button(
        "1 · Transcribe", type="primary", disabled=not ordered, use_container_width=True
    ):
        run_transcription(ordered, settings)

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
        questions_tab(settings, review)
    with t4:
        export_tab()


if __name__ == "__main__":
    main()
