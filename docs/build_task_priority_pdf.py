#!/usr/bin/env python3
"""Build the Task Priority Kinematic Control deep-dive PDF from Markdown.

This deliberately uses ReportLab because the workspace does not require pandoc
or a LaTeX installation. The renderer is intentionally simple: headings,
paragraphs, bullets, tables, and fenced code blocks are enough for this document.
"""

from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "task_priority_kinematic_control_deep_dive.md"
OUTPUT = ROOT / "task_priority_kinematic_control_deep_dive.pdf"


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            spaceAfter=5,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.3,
            leading=9,
            leftIndent=4,
            rightIndent=4,
            spaceBefore=4,
            spaceAfter=7,
            backColor=colors.HexColor("#f6f8fa"),
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=colors.white,
        ),
    }
    return styles


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    out = ""
    in_code = False
    token = ""
    for char in escaped:
        if char == "`":
            if in_code:
                out += f'<font name="Courier">{token}</font>'
                token = ""
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            token += char
        else:
            out += char
    if in_code:
        out += f"`{token}"
    return out


def table_from_rows(rows, styles):
    data = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell.strip()), style) for cell in row])
    col_count = max(len(row) for row in data)
    usable_width = A4[0] - 3.2 * cm
    col_widths = [usable_width / col_count] * col_count
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0bec5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def parse_markdown(text: str):
    styles = make_styles()
    story = []
    lines = text.splitlines()
    i = 0
    pending_para = []

    def flush_para():
        nonlocal pending_para
        if pending_para:
            story.append(Paragraph(inline_markup(" ".join(pending_para)), styles["body"]))
            pending_para = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(block), styles["code"], maxLineLength=95))
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip() for c in raw.split("|")]
                if not all(set(c) <= {"-", ":"} for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                story.append(table_from_rows(rows, styles))
                story.append(Spacer(1, 6))
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_para()
            story.append(Paragraph(inline_markup(stripped[2:].strip()), styles["title"]))
            story.append(Spacer(1, 6))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_para()
            story.append(Paragraph(inline_markup(stripped[3:].strip()), styles["h1"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_para()
            story.append(Paragraph(inline_markup(stripped[4:].strip()), styles["h2"]))
            i += 1
            continue

        if stripped.startswith("#### "):
            flush_para()
            story.append(Paragraph(inline_markup(stripped[5:].strip()), styles["h3"]))
            i += 1
            continue

        if stripped.startswith("- "):
            flush_para()
            story.append(Paragraph(f"• {inline_markup(stripped[2:].strip())}", styles["bullet"]))
            i += 1
            continue

        if len(stripped) > 2 and stripped[0].isdigit() and ". " in stripped[:4]:
            flush_para()
            story.append(Paragraph(inline_markup(stripped), styles["bullet"]))
            i += 1
            continue

        pending_para.append(stripped)
        i += 1

    flush_para()
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607d8b"))
    text = f"Task Priority Kinematic Control Deep Dive - page {doc.page}"
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, text)
    canvas.restoreState()


def main():
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"))
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Task Priority Kinematic Control Deep Dive",
        author="OpenAI Codex",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
