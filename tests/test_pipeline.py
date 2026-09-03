"""Tests for the parts that must not silently break: schema, validation,
answer balancing, chunk allocation, and every exporter's output being a
well-formed file of the type it claims to be.

Run: pytest -q
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

import pytest

from src.chunking import allocate_questions, chunk_transcript
from src.config import DEFAULT_PROVIDER, PROVIDERS, AppSettings, get_provider
from src.exporters import (
    export_csv,
    export_docx,
    export_markdown,
    export_pdf,
    export_qti12_canvas,
    export_qti21,
    export_xlsx,
)
from src.exporters.transcript_formats import export_srt, export_vtt
from src.llm import PRICING, Usage, estimate_cost, has_pricing, parse_json_object
from src.mcq import (
    _coerce_mcq,
    balance_answer_positions,
    validate_all,
    validate_question,
)
from src.schema import MCQ, Quiz, QuizMeta, Segment, Transcript, format_timestamp, parse_timestamp


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def transcript() -> Transcript:
    segments = [
        Segment(index=i, start=i * 30.0, end=(i + 1) * 30.0,
                text=f"This is segment {i} discussing operations topic {i}. " * 6)
        for i in range(40)  # 20 minutes
    ]
    return Transcript(segments=segments, language="en", duration=1200.0)


def make_q(**kwargs) -> MCQ:
    base = dict(
        stem="Which planning strategy varies workforce levels to match demand?",
        options=["Chase strategy", "Level strategy", "Mixed strategy", "Fixed strategy"],
        correct_index=0,
        rationale="A chase strategy adjusts capacity to track demand period by period.",
        distractor_rationales=["", "confuses level with chase", "partial", "not a strategy"],
        bloom="Understand",
        difficulty="Medium",
        topic="Aggregate planning",
        source_timestamp="4:20",
        source_quote="the chase strategy hires and lays off to match demand",
    )
    base.update(kwargs)
    return MCQ(**base)


@pytest.fixture
def quiz() -> Quiz:
    questions = [
        make_q(stem=f"Question number {i} about operations planning concepts?",
               correct_index=i % 4,
               topic=f"Topic {i}")
        for i in range(8)
    ]
    return Quiz(meta=QuizMeta(title="Ops Lecture 4", course="MGT 301",
                              generated_on="2026-09-02"), questions=questions)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_timestamp_roundtrip():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(75) == "1:15"
    assert format_timestamp(3725) == "1:02:05"
    assert parse_timestamp("1:02:05") == 3725
    assert parse_timestamp("4:20") == 260
    assert parse_timestamp("") == 0.0


def test_mcq_rejects_out_of_range_answer():
    with pytest.raises(Exception):
        MCQ(stem="x" * 20, options=["a", "b", "c"], correct_index=5)


def test_mcq_rejects_too_few_options():
    with pytest.raises(Exception):
        MCQ(stem="x" * 20, options=["a", "b"], correct_index=0)


def test_transcript_text_and_slice(transcript):
    assert transcript.word_count > 100
    assert "[0:00]" in transcript.text_with_timestamps()
    assert transcript.slice_by_time(0, 60).strip()


# --------------------------------------------------------------------------- #
# Provider registry
# --------------------------------------------------------------------------- #


def test_gemini_is_the_default_provider():
    assert DEFAULT_PROVIDER == "gemini"
    assert list(PROVIDERS)[0] == "gemini", "default should be first in the sidebar"
    assert AppSettings().provider == "gemini"
    assert AppSettings().llm_model in PROVIDERS["gemini"].models


def test_all_five_providers_are_registered():
    assert set(PROVIDERS) == {"gemini", "anthropic", "openai", "xai", "openrouter"}


def test_every_provider_is_completely_specified():
    for key, spec in PROVIDERS.items():
        assert spec.key == key
        assert spec.sdk in {"gemini", "anthropic", "openai"}
        assert spec.env_var.endswith("_API_KEY")
        assert spec.models, f"{key} has no models"
        assert spec.console_url.startswith("https://")


def test_openai_compatible_providers_have_distinct_base_urls():
    xai = PROVIDERS["xai"]
    router = PROVIDERS["openrouter"]
    assert xai.sdk == router.sdk == "openai"
    assert xai.base_url == "https://api.x.ai/v1"
    assert router.base_url == "https://openrouter.ai/api/v1"
    assert PROVIDERS["openai"].base_url is None  # SDK default


def test_openrouter_allows_custom_slugs_and_skips_native_json_mode():
    router = PROVIDERS["openrouter"]
    assert router.allow_custom_model is True
    assert router.supports_json_mode is False
    assert all("/" in m for m in router.models), "OpenRouter slugs are vendor/model"


def test_get_provider_falls_back_to_the_default():
    assert get_provider("nonsense").key == DEFAULT_PROVIDER
    assert get_provider("xai").label == "xAI Grok"


def test_api_key_resolution_prefers_explicit_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert AppSettings(provider="gemini").resolved_api_key() == "from-env"
    assert AppSettings(provider="gemini", api_key="typed").resolved_api_key() == "typed"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert AppSettings(provider="gemini").resolved_api_key() == ""


def test_each_provider_reads_its_own_env_var(monkeypatch):
    for key, spec in PROVIDERS.items():
        monkeypatch.setenv(spec.env_var, f"key-for-{key}")
    for key, spec in PROVIDERS.items():
        assert AppSettings(provider=key).resolved_api_key() == f"key-for-{key}"


def test_pricing_covers_every_listed_model_except_openrouter():
    for key, spec in PROVIDERS.items():
        for model in spec.models:
            if key == "openrouter":
                continue  # priced per underlying model upstream
            assert model in PRICING, f"{model} missing from PRICING"


def test_cost_estimation_and_unpriced_models():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("gemini-3.8-flash", usage) == pytest.approx(0.75 + 3.75)
    assert has_pricing("grok-4.6")
    assert not has_pricing("meta-llama/llama-4-maverick")
    assert estimate_cost("meta-llama/llama-4-maverick", usage) == 0.0


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_chunking_covers_full_duration(transcript):
    chunks = chunk_transcript(transcript, chunk_seconds=300, overlap_seconds=30)
    assert len(chunks) >= 4
    assert chunks[0].start == 0.0
    assert chunks[-1].end == pytest.approx(transcript.duration, abs=1.0)
    assert all(c.text for c in chunks)


def test_short_transcript_is_one_chunk(transcript):
    chunks = chunk_transcript(transcript, chunk_seconds=3600)
    assert len(chunks) == 1


def test_allocation_sums_to_target():
    assert sum(allocate_questions(10, 3)) == 10
    assert allocate_questions(10, 3) == [4, 3, 3]
    assert allocate_questions(2, 5) == [1, 1, 0, 0, 0]
    assert allocate_questions(5, 0) == []


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_clean_question_has_no_flags():
    assert validate_question(make_q()) == []


def test_flags_all_of_the_above():
    q = make_q(options=["Chase", "Level", "Mixed", "All of the above"])
    assert any("all/none" in f for f in validate_question(q))


def test_flags_duplicate_options():
    q = make_q(options=["Chase strategy", "chase strategy!", "Level", "Mixed"])
    assert any("duplicate" in f.lower() for f in validate_question(q))


def test_flags_overlong_correct_answer():
    q = make_q(
        options=[
            "A chase strategy adjusts the size of the workforce each period so that "
            "production capacity tracks the forecast demand as closely as possible",
            "Level", "Mixed", "Fixed",
        ],
        correct_index=0,
    )
    assert any("longer" in f for f in validate_question(q))


def test_flags_negative_stem():
    q = make_q(stem="Which of the following is NOT an aggregate planning strategy?")
    assert any("Negatively phrased" in f for f in validate_question(q))


def test_flags_missing_provenance():
    q = make_q(source_quote="", rationale="")
    flags = validate_question(q)
    assert any("supporting quote" in f for f in flags)
    assert any("rationale" in f.lower() for f in flags)


def test_cross_item_duplicate_detection():
    a = make_q(stem="Which planning strategy varies workforce to match demand?")
    b = make_q(stem="Which strategy varies the workforce to match demand planning?")
    flagged = validate_all([a, b])
    assert any("duplicate" in f.lower() for q in flagged for f in q.flags)


# --------------------------------------------------------------------------- #
# Answer balancing
# --------------------------------------------------------------------------- #


def test_balancing_spreads_answers_and_preserves_correctness():
    questions = [make_q(stem=f"Stem {i} about planning strategies?", correct_index=0)
                 for i in range(12)]
    originals = [q.correct_option for q in questions]

    balanced = balance_answer_positions(questions, seed=3)

    for q, original in zip(balanced, originals):
        assert q.correct_option == original, "balancing must not change the right answer"

    counts = Counter(q.answer_letter for q in balanced)
    assert len(counts) == 4, "answers should land in every position"
    assert max(counts.values()) <= 4


def test_balancing_moves_rationales_with_options():
    q = make_q(correct_index=0,
               distractor_rationales=["correct", "d1", "d2", "d3"])
    balance_answer_positions([q], seed=1)
    assert q.distractor_rationales[q.correct_index] == "correct"


# --------------------------------------------------------------------------- #
# Model-output coercion
# --------------------------------------------------------------------------- #


def test_coerce_accepts_letter_answers():
    item = _coerce_mcq(
        {"stem": "What does a level strategy hold constant?",
         "options": ["Output rate", "Demand", "Price", "Lead time"],
         "correct_index": "B"}
    )
    assert item is not None and item.correct_index == 1


def test_coerce_rejects_unusable_payloads():
    assert _coerce_mcq({"stem": "", "options": ["a", "b", "c"]}) is None
    assert _coerce_mcq({"stem": "ok stem here", "options": ["a"]}) is None
    assert _coerce_mcq({"stem": "ok stem here", "options": ["a", "b", "c"],
                        "correct_index": 9}) is None


def test_coerce_normalizes_bad_enum_values():
    item = _coerce_mcq(
        {"stem": "A reasonable stem about planning?",
         "options": ["a", "b", "c", "d"], "correct_index": 0,
         "bloom": "synthesis", "difficulty": "brutal"}
    )
    assert item.bloom == "Understand" and item.difficulty == "Medium"


def test_json_salvage_from_fenced_and_noisy_output():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": [1,2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(Exception):
        parse_json_object("no json at all")


# --------------------------------------------------------------------------- #
# Exporters
# --------------------------------------------------------------------------- #


def test_csv_has_a_row_per_question(quiz):
    text = export_csv(quiz).decode("utf-8-sig")
    assert text.count("\n") >= len(quiz.included)
    assert "Correct" in text.splitlines()[0]


def test_xlsx_is_a_valid_workbook(quiz):
    data = export_xlsx(quiz)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    assert "xl/workbook.xml" in names


def test_docx_is_a_valid_package(quiz):
    with zipfile.ZipFile(io.BytesIO(export_docx(quiz))) as zf:
        assert "word/document.xml" in zf.namelist()


def test_pdf_has_a_pdf_header(quiz):
    data = export_pdf(quiz)
    assert data[:5] == b"%PDF-"
    assert len(data) > 1500


def test_markdown_contains_answer_key(quiz):
    text = export_markdown(quiz).decode("utf-8")
    assert "## Answer key" in text
    assert text.count("- A.") == len(quiz.included)


def test_qti12_package_parses_and_has_one_item_per_question(quiz):
    with zipfile.ZipFile(io.BytesIO(export_qti12_canvas(quiz))) as zf:
        names = zf.namelist()
        assert "imsmanifest.xml" in names
        ET.fromstring(zf.read("imsmanifest.xml"))

        assessment_name = next(n for n in names if n.endswith(".xml") and "/" in n
                               and not n.endswith("assessment_meta.xml"))
        root = ET.fromstring(zf.read(assessment_name))
        ns = "{http://www.imsglobal.org/xsd/ims_qtiasiv1p2}"
        items = root.findall(f".//{ns}item")
        assert len(items) == len(quiz.included)

        meta_name = next(n for n in names if n.endswith("assessment_meta.xml"))
        ET.fromstring(zf.read(meta_name))


def test_qti12_marks_exactly_one_correct_response(quiz):
    with zipfile.ZipFile(io.BytesIO(export_qti12_canvas(quiz))) as zf:
        name = next(n for n in zf.namelist() if "/" in n and n.endswith(".xml")
                    and not n.endswith("assessment_meta.xml"))
        root = ET.fromstring(zf.read(name))
    ns = "{http://www.imsglobal.org/xsd/ims_qtiasiv1p2}"
    for item in root.findall(f".//{ns}item"):
        scoring = [
            c for c in item.findall(f".//{ns}respcondition")
            if c.find(f"{ns}setvar") is not None
        ]
        assert len(scoring) == 1


def test_qti21_items_parse_and_reference_a_real_choice(quiz):
    with zipfile.ZipFile(io.BytesIO(export_qti21(quiz))) as zf:
        names = zf.namelist()
        assert "imsmanifest.xml" in names
        item_files = [n for n in names if n.startswith("items/")]
        assert len(item_files) == len(quiz.included)

        ns = "{http://www.imsglobal.org/xsd/imsqti_v2p1}"
        for name in item_files:
            root = ET.fromstring(zf.read(name))
            correct = root.find(f".//{ns}correctResponse/{ns}value").text
            ids = [c.get("identifier") for c in root.findall(f".//{ns}simpleChoice")]
            assert correct in ids


def test_exports_respect_the_include_flag(quiz):
    quiz.questions[0].include = False
    assert len(quiz.included) == len(quiz.questions) - 1
    with zipfile.ZipFile(io.BytesIO(export_qti21(quiz))) as zf:
        assert len([n for n in zf.namelist() if n.startswith("items/")]) == len(quiz.included)


def test_xml_special_characters_survive_export():
    q = make_q(
        stem="If cost < revenue & margin > 0, which statement holds?",
        options=['A "quoted" claim', "B & C", "<tagged>", "plain"],
        correct_index=1,
    )
    quiz = Quiz(meta=QuizMeta(title="Edge & Cases"), questions=[q])
    with zipfile.ZipFile(io.BytesIO(export_qti12_canvas(quiz))) as zf:
        for name in zf.namelist():
            ET.fromstring(zf.read(name))  # must not raise
    with zipfile.ZipFile(io.BytesIO(export_qti21(quiz))) as zf:
        for name in zf.namelist():
            ET.fromstring(zf.read(name))
    assert export_pdf(quiz)[:5] == b"%PDF-"


# --------------------------------------------------------------------------- #
# Caption exports
# --------------------------------------------------------------------------- #


def test_srt_and_vtt_shape(transcript):
    srt = export_srt(transcript).decode("utf-8")
    assert srt.startswith("1\n")
    assert "00:00:00,000 --> 00:00:30,000" in srt

    vtt = export_vtt(transcript).decode("utf-8")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:30.000" in vtt
