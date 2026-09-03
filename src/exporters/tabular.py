"""CSV and XLSX exports.

The CSV column layout deliberately matches the shape most LMS bulk-import
templates and question-bank tools expect: stem, then one column per option,
then the answer letter. The extra pedagogical columns sit to the right, where
they can be deleted in one selection if a strict importer objects.
"""

from __future__ import annotations

import io

import pandas as pd

from ..schema import OPTION_LETTERS, Quiz


def quiz_to_dataframe(quiz: Quiz) -> pd.DataFrame:
    max_options = max((len(q.options) for q in quiz.included), default=4)
    rows: list[dict] = []

    for i, q in enumerate(quiz.included, start=1):
        row: dict[str, object] = {
            "No": i,
            "Question": q.stem,
        }
        for j in range(max_options):
            row[f"Option {OPTION_LETTERS[j]}"] = q.options[j] if j < len(q.options) else ""
        row.update(
            {
                "Correct": q.answer_letter,
                "Correct Text": q.correct_option,
                "Points": q.points,
                "Bloom": q.bloom,
                "Difficulty": q.difficulty,
                "Topic": q.topic,
                "Rationale": q.rationale,
                "Timestamp": q.source_timestamp,
                "Source Quote": q.source_quote,
                "Review Flags": "; ".join(q.flags),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def export_csv(quiz: Quiz) -> bytes:
    return quiz_to_dataframe(quiz).to_csv(index=False).encode("utf-8-sig")


def export_xlsx(quiz: Quiz) -> bytes:
    """Workbook with a questions sheet, an answer key, and a coverage sheet."""
    from ..mcq import coverage_report

    df = quiz_to_dataframe(quiz)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Questions", index=False, startrow=1, header=False)
        book = writer.book
        sheet = writer.sheets["Questions"]

        header_fmt = book.add_format(
            {"bold": True, "bg_color": "#00539B", "font_color": "white",
             "border": 1, "text_wrap": True, "valign": "vcenter"}
        )
        wrap = book.add_format({"text_wrap": True, "valign": "top"})

        for col, name in enumerate(df.columns):
            sheet.write(0, col, name, header_fmt)

        widths = {"Question": 60, "Rationale": 50, "Source Quote": 45, "Review Flags": 35}
        for col, name in enumerate(df.columns):
            sheet.set_column(col, col, widths.get(name, 22 if "Option" in str(name) else 14), wrap)
        sheet.freeze_panes(1, 2)
        sheet.autofilter(0, 0, max(1, len(df)), len(df.columns) - 1)

        key = pd.DataFrame(
            {
                "No": range(1, len(quiz.included) + 1),
                "Answer": [q.answer_letter for q in quiz.included],
                "Points": [q.points for q in quiz.included],
                "Topic": [q.topic for q in quiz.included],
            }
        )
        key.to_excel(writer, sheet_name="Answer Key", index=False)

        cov = coverage_report(quiz.included)
        cov_rows = [
            {"Dimension": dim, "Value": k, "Count": v}
            for dim, counts in cov.items()
            for k, v in sorted(counts.items())
        ]
        pd.DataFrame(cov_rows).to_excel(writer, sheet_name="Coverage", index=False)

        meta = pd.DataFrame(
            [
                {"Field": "Title", "Value": quiz.meta.title},
                {"Field": "Course", "Value": quiz.meta.course},
                {"Field": "Source file", "Value": quiz.meta.source_filename},
                {"Field": "Generated", "Value": quiz.meta.generated_on},
                {"Field": "Model", "Value": quiz.meta.model_used},
                {"Field": "Questions", "Value": len(quiz.included)},
                {"Field": "Total points", "Value": quiz.total_points},
            ]
        )
        meta.to_excel(writer, sheet_name="About", index=False)

    return buffer.getvalue()
