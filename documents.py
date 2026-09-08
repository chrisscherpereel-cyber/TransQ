"""Printable outputs: Word worksheet, PDF worksheet, and Markdown.

Each document is a student-facing quiz followed by an instructor answer key on
its own page, so the same file can be printed and then split.
"""

from __future__ import annotations

import io

from ..schema import Quiz, Summary


# --------------------------------------------------------------------------- #
# Word
# --------------------------------------------------------------------------- #


def export_docx(quiz: Quiz, summary: Summary | None = None, include_key: bool = True) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.shared import Pt, RGBColor

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_heading(quiz.meta.title or "Lecture Quiz", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub_bits = [b for b in (quiz.meta.course, quiz.meta.generated_on) if b]
    if sub_bits:
        sub = doc.add_paragraph(" · ".join(sub_bits))
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub.runs[0].font.size = Pt(9)
        sub.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    info = doc.add_paragraph()
    info.add_run("Name: ").bold = True
    info.add_run("_" * 34 + "    ")
    info.add_run("Date: ").bold = True
    info.add_run("_" * 18)
    doc.add_paragraph(
        f"{len(quiz.included)} questions · {quiz.total_points:g} points total"
    ).runs[0].font.size = Pt(9)

    if summary and summary.abstract:
        doc.add_heading("Lecture summary", level=1)
        doc.add_paragraph(summary.abstract)
        if summary.learning_objectives:
            doc.add_heading("Learning objectives", level=2)
            for obj in summary.learning_objectives:
                doc.add_paragraph(obj, style="List Bullet")

    doc.add_heading("Questions", level=1)
    doc.add_paragraph(
        "Select the single best answer for each question."
    ).runs[0].italic = True

    for i, q in enumerate(quiz.included, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.add_run(f"{i}. ").bold = True
        p.add_run(q.stem)
        for letter, opt in q.lettered_options():
            op = doc.add_paragraph()
            op.paragraph_format.left_indent = Pt(28)
            op.paragraph_format.space_after = Pt(2)
            op.add_run(f"{letter}. ").bold = True
            op.add_run(opt)

    if include_key:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        doc.add_heading("Answer key — instructor copy", level=1)

        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        for cell, header in zip(
            table.rows[0].cells, ["#", "Ans", "Bloom", "Timestamp", "Rationale"]
        ):
            cell.text = header
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

        for i, q in enumerate(quiz.included, start=1):
            cells = table.add_row().cells
            cells[0].text = str(i)
            cells[1].text = q.answer_letter
            cells[2].text = q.bloom
            cells[3].text = q.source_timestamp
            cells[4].text = q.rationale

        flagged = [(i, q) for i, q in enumerate(quiz.included, start=1) if q.flags]
        if flagged:
            doc.add_heading("Items to review before use", level=2)
            for i, q in flagged:
                doc.add_paragraph(f"Q{i}: {'; '.join(q.flags)}", style="List Bullet")

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def export_pdf(quiz: Quiz, summary: Summary | None = None, include_key: bool = True) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    def esc(text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=quiz.meta.title or "Lecture Quiz",
    )

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceAfter=4,
                        textColor=colors.HexColor("#00539B"))
    center = ParagraphStyle("Center", parent=ss["Normal"], alignment=TA_CENTER,
                            fontSize=9, textColor=colors.HexColor("#6B7280"))
    body = ParagraphStyle("Body", parent=ss["Normal"], fontSize=10.5, leading=14.5)
    stem = ParagraphStyle("Stem", parent=body, spaceBefore=9, spaceAfter=3)
    option = ParagraphStyle("Option", parent=body, leftIndent=22, spaceAfter=1.5)
    h2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=12)

    flow: list = [Paragraph(esc(quiz.meta.title or "Lecture Quiz"), h1)]
    sub_bits = [b for b in (quiz.meta.course, quiz.meta.generated_on) if b]
    if sub_bits:
        flow.append(Paragraph(esc(" · ".join(sub_bits)), center))
    flow.append(Spacer(1, 10))
    flow.append(Paragraph("Name: " + "_" * 40 + "&nbsp;&nbsp;Date: " + "_" * 16, body))
    flow.append(
        Paragraph(
            f"{len(quiz.included)} questions &middot; {quiz.total_points:g} points total",
            center,
        )
    )
    flow.append(Spacer(1, 8))

    if summary and summary.abstract:
        flow.append(Paragraph("Lecture summary", h2))
        flow.append(Paragraph(esc(summary.abstract), body))
        if summary.learning_objectives:
            flow.append(Paragraph("Learning objectives", h2))
            for obj in summary.learning_objectives:
                flow.append(Paragraph(f"&bull; {esc(obj)}", option))

    flow.append(Paragraph("Questions", h2))
    flow.append(Paragraph("<i>Select the single best answer for each question.</i>", body))

    for i, q in enumerate(quiz.included, start=1):
        flow.append(Paragraph(f"<b>{i}.</b> {esc(q.stem)}", stem))
        for letter, opt in q.lettered_options():
            flow.append(Paragraph(f"<b>{letter}.</b> {esc(opt)}", option))

    if include_key:
        flow.append(PageBreak())
        flow.append(Paragraph("Answer key — instructor copy", h1))
        flow.append(Spacer(1, 8))

        data = [["#", "Ans", "Bloom", "Time", "Rationale"]]
        cell = ParagraphStyle("Cell", parent=body, fontSize=8.5, leading=11)
        for i, q in enumerate(quiz.included, start=1):
            data.append(
                [
                    str(i),
                    q.answer_letter,
                    q.bloom,
                    q.source_timestamp,
                    Paragraph(esc(q.rationale), cell),
                ]
            )

        table = Table(data, colWidths=[24, 30, 62, 46, 288], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00539B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D5DD")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#F6F8FA")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        flow.append(table)

        flagged = [(i, q) for i, q in enumerate(quiz.included, start=1) if q.flags]
        if flagged:
            flow.append(Paragraph("Items to review before use", h2))
            for i, q in flagged:
                flow.append(Paragraph(f"&bull; <b>Q{i}:</b> {esc('; '.join(q.flags))}", option))

    doc.build(flow)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def export_markdown(quiz: Quiz, summary: Summary | None = None) -> bytes:
    lines: list[str] = [f"# {quiz.meta.title or 'Lecture Quiz'}", ""]
    if quiz.meta.course:
        lines += [f"*{quiz.meta.course}*", ""]
    if summary:
        lines += [summary.as_markdown(), ""]

    lines += ["## Questions", ""]
    for i, q in enumerate(quiz.included, start=1):
        lines.append(f"**{i}. {q.stem}**")
        lines.append("")
        for letter, opt in q.lettered_options():
            lines.append(f"- {letter}. {opt}")
        lines.append("")

    lines += ["", "---", "", "## Answer key", ""]
    for i, q in enumerate(quiz.included, start=1):
        lines.append(f"{i}. **{q.answer_letter}** — {q.rationale} *(from {q.source_timestamp})*")

    return "\n".join(lines).encode("utf-8")
