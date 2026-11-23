from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from .schemas import Source
def safe_filename_stem(title: str, fallback: str, max_length: int = 80) -> str:
    """Turn a research title into a filesystem- and header-safe filename stem.
    Downloads were previously named after the internal run id (a random hex
    string), which is unreadable and inconsistent between a report's PDF,
    DOCX, LaTeX, and BibTeX files. This normalizes the *readable* title into
    something safe to put in a filename and an HTTP header, without
    unnecessarily mangling an ordinary title -- and only falls back to
    `fallback` (the run id) when the title is missing or empty after
    normalization, so a real title always wins over a random id.
    """
    text = unicodedata.normalize("NFKC", (title or "").strip())
    text = re.sub(r'[\\/:*?"<>|\r\n\t\x00-\x1f]', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return fallback
    if len(text) > max_length:
        truncated = text[:max_length].rsplit(" ", 1)[0].strip(" .")
        text = truncated or text[:max_length].strip(" .")
    return text or fallback
def content_disposition_header(filename_stem: str, extension: str) -> str:
    """Build a `Content-Disposition` value safe for ASCII-only clients while
    preserving a non-ASCII title for clients that honor RFC 5987/6266's
    extended `filename*=` syntax (most current browsers)."""
    ascii_stem = filename_stem.encode("ascii", "ignore").decode("ascii").strip(" ._-") or "report"
    ascii_name = f"{ascii_stem}.{extension}"
    utf8_name = quote(f"{filename_stem}.{extension}")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
def _strip_bold(text: str) -> str:
    return _BOLD_RE.sub(r"\1", text)
@dataclass
class _Block:
    kind: str
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|?$")
def _split_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [_strip_bold(c.strip()) for c in cells]
def parse_blocks(markdown: str) -> list[_Block]:
    lines = markdown.splitlines()
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()
        if not line:
            i += 1
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(_Block("h1", _strip_bold(line[2:])))
        elif line.startswith("### "):
            blocks.append(_Block("h3", _strip_bold(line[4:])))
        elif line.startswith("## "):
            blocks.append(_Block("h2", _strip_bold(line[3:])))
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            blocks.append(_Block("meta", _strip_bold(line.strip("*"))))
        elif line == "---":
            blocks.append(_Block("hr"))
        elif line.startswith("> "):
            blocks.append(_Block("quote", _strip_bold(line[2:]).strip("\u201c\u201d")))
        elif line.startswith("- ") or line.startswith("  - "):
            blocks.append(_Block("bullet", _strip_bold(line.lstrip(" -"))))
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1
            rows = [_split_table_row(t) for t in table_lines if not _TABLE_SEP_RE.match(t)]
            if rows:
                blocks.append(_Block("table", rows=rows))
        else:
            blocks.append(_Block("para", _strip_bold(line)))
        i += 1
    return blocks
_CITEKEY_RE = re.compile(r"[^a-zA-Z0-9]")
def _bibtex_citekey(source: Source, used_keys: set[str]) -> str:
    first_author = (source.authors.split(",")[0].split(" ")[-1] if source.authors else "anon").lower()
    first_author = _CITEKEY_RE.sub("", first_author) or "anon"
    year = source.year or "n_d"
    base = f"{first_author}{year}"
    key = base
    suffix = ord("a")
    while key in used_keys:
        key = f"{base}{chr(suffix)}"
        suffix += 1
    used_keys.add(key)
    return key
def _bibtex_escape(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("\\", "")
def sources_to_bibtex(sources: list[Source]) -> str:
    """Export selected sources as a .bib file so they can be dropped
    straight into a LaTeX/Overleaf project or reference manager."""
    used_keys: set[str] = set()
    entries = []
    for source in sources:
        key = _bibtex_citekey(source, used_keys)
        entry_type = "article" if source.doi else "misc"
        fields = [
            f'  title = {{{_bibtex_escape(source.title)}}}',
        ]
        if source.authors:
            fields.append(f'  author = {{{_bibtex_escape(source.authors)}}}')
        if source.year:
            fields.append(f'  year = {{{source.year}}}')
        if source.venue:
            fields.append(f'  journal = {{{_bibtex_escape(source.venue)}}}')
        if source.doi:
            fields.append(f'  doi = {{{source.doi}}}')
        url = source.url or source.pdf_url
        if url:
            fields.append(f'  url = {{{url}}}')
        fields.append(f'  note = {{Retrieved via {source.provider}}}')
        entries.append(f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + "\n"
def _set_cell_shading(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)
def _set_cell_borders(cell, color: str = "B8B8B8", size: str = "4") -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        el = borders.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:color"), color)
def _set_repeat_table_header(row) -> None:
    """Make a table header repeat when a table spans pages in Word."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
def _set_row_cant_split(row) -> None:
    """Keep short table rows together instead of splitting a row across pages."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    trPr = row._tr.get_or_add_trPr()
    cantSplit = OxmlElement("w:cantSplit")
    cantSplit.set(qn("w:val"), "true")
    trPr.append(cantSplit)
def _set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcMar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
def _add_page_number(paragraph) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
def _add_inline_runs(paragraph, text: str, normal_font: str = "Georgia", citation_color=None) -> None:
    """Render the report's small inline-markdown subset with readable
    academic body text and visually distinct source citations."""
    import re
    from docx.shared import Pt, RGBColor
    pattern = re.compile(r"(\*\*[^*]+\*\*|\[[SE]\d+(?:,\s*p\.\s*\d+)?\]|\*[^*]+\*)")
    pos = 0
    for match in pattern.finditer(str(text or "")):
        if match.start() > pos:
            run = paragraph.add_run(text[pos:match.start()])
            run.font.name = normal_font; run.font.size = Pt(10.75)
        token = match.group(0)
        run = paragraph.add_run(token.strip("*"))
        run.font.name = normal_font; run.font.size = Pt(10.75)
        if token.startswith("["):
            run.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = citation_color or RGBColor(0x4F, 0x74, 0x5D)
        elif token.startswith("**"):
            run.bold = True
        else:
            run.italic = True
        pos = match.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.font.name = normal_font; run.font.size = Pt(10.75)
def markdown_to_docx(markdown: str, out_path: str | Path) -> Path:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml.ns import qn
    blocks = parse_blocks(markdown)
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Georgia"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Georgia") if normal._element.rPr is not None else None
    normal.font.size = Pt(10.75)
    normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.22
    normal.paragraph_format.widow_control = True
    for name, size, bold, color in [
        ("Title", 26, True, "26302B"),
        ("Heading 1", 18, True, "26302B"),
        ("Heading 2", 14, True, "3F5548"),
        ("Heading 3", 11.5, True, "59655E"),
    ]:
        st = styles[name]
        st.font.name = "Georgia"
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(12 if name != "Title" else 0)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
    report_title = _first_h1(blocks) or "Research Report"
    header = section.header.paragraphs[0]
    header.text = report_title[:100]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("Research report  ·  ")
    fr.font.name = "Aptos"
    fr.font.size = Pt(8)
    fr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    _add_page_number(footer)
    for block in blocks:
        if block.kind == "h1":
            heading = doc.add_paragraph()
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
            heading.paragraph_format.space_before = Pt(0)
            heading.paragraph_format.space_after = Pt(4)
            heading.paragraph_format.keep_with_next = True
            run = heading.add_run(block.text)
            run.font.name = "Aptos Display"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
        elif block.kind == "meta":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(14)
            run = p.add_run(block.text)
            run.italic = True
            run.font.name = "Aptos"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        elif block.kind == "hr":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
        elif block.kind == "h2":
            heading = doc.add_paragraph()
            heading.paragraph_format.space_before = Pt(13)
            heading.paragraph_format.space_after = Pt(5)
            heading.paragraph_format.keep_with_next = True
            run = heading.add_run(block.text)
            run.font.name = "Aptos"
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
        elif block.kind == "h3":
            heading = doc.add_paragraph()
            heading.paragraph_format.space_before = Pt(9)
            heading.paragraph_format.space_after = Pt(4)
            heading.paragraph_format.keep_with_next = True
            run = heading.add_run(block.text)
            run.font.name = "Aptos"
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        elif block.kind == "bullet":
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            _add_inline_runs(p, block.text, normal_font="Georgia", citation_color=RGBColor(0x4F,0x74,0x5D))
        elif block.kind == "quote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.28)
            p.paragraph_format.right_indent = Inches(0.15)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(f'“{block.text}”')
            run.italic = True
            run.font.name = "Georgia"
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(0x55, 0x5E, 0x58)
        elif block.kind == "table" and block.rows:
            n_cols = max(len(r) for r in block.rows)
            table = doc.add_table(rows=0, cols=n_cols)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True
            for r_idx, row in enumerate(block.rows):
                row_obj = table.add_row()
                _set_row_cant_split(row_obj)
                if r_idx == 0:
                    _set_repeat_table_header(row_obj)
                cells = row_obj.cells
                for c_idx in range(n_cols):
                    cell = cells[c_idx]
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    cell.text = row[c_idx] if c_idx < len(row) else ""
                    _set_cell_margins(cell)
                    _set_cell_borders(cell)
                    if r_idx == 0:
                        _set_cell_shading(cell, "333333")
                    else:
                        _set_cell_shading(cell, "FFFFFF")
                    for para in cell.paragraphs:
                        para.paragraph_format.space_after = Pt(2)
                        for run in para.runs:
                            run.font.name = "Aptos"
                            run.font.size = Pt(9.0)
                            if r_idx == 0:
                                run.bold = True
                                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                            else:
                                run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif block.kind == "para":
            p = doc.add_paragraph()
            p.paragraph_format.widow_control = True
            p.paragraph_format.keep_together = False
            _add_inline_runs(p, block.text, normal_font="Georgia", citation_color=RGBColor(0x4F,0x74,0x5D))
    doc.core_properties.title = _first_h1(blocks) or "Research Report"
    doc.core_properties.subject = "Evidence-grounded research report"
    doc.core_properties.author = ""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return out_path
def _first_h1(blocks: list[_Block]) -> str:
    for block in blocks:
        if block.kind == "h1":
            return block.text
    return ""
def _convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> bool:
    """Prefer a real office renderer over a hand-drawn PDF.
    Order: Microsoft Word on Windows, then LibreOffice/soffice. Both render the
    already-polished DOCX rather than maintaining a second report layout. The
    Windows path is preferred because the application is primarily distributed
    to desktop researchers who already have Word. A false return lets the caller
    use the environment-safe fallback.
    """
    import shutil
    import subprocess
    import sys
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        word = None
        doc = None
        try:
            import win32com.client
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(docx_path.resolve()))
            doc.ExportAsFixedFormat(str(pdf_path.resolve()), 17)
            if pdf_path.exists():
                return True
        except Exception:
            pass
        finally:
            try:
                if doc is not None:
                    doc.Close(False)
            except Exception:
                pass
            try:
                if word is not None:
                    word.Quit()
            except Exception:
                pass
    for exe in ("soffice", "libreoffice"):
        binary = shutil.which(exe)
        if not binary:
            continue
        try:
            subprocess.run([binary, "--headless", "--convert-to", "pdf", "--outdir", str(pdf_path.parent), str(docx_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            generated = pdf_path.parent / (docx_path.stem + ".pdf")
            if generated.exists():
                if generated != pdf_path:
                    generated.replace(pdf_path)
                return True
        except Exception:
            pass
    return False
def markdown_to_pdf(markdown: str, out_path: str | Path) -> Path:
    """Render a PDF reliably in both desktop and server environments.
    The polished DOCX is the primary renderer. If Word/LibreOffice is absent,
    a ReportLab fallback guarantees that the API still returns a real PDF
    rather than a 500. The fallback shares the same parsed Markdown blocks and
    is deliberately conservative about layout.
    """
    from tempfile import TemporaryDirectory
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="researchpilot_export_") as tmp:
        docx_path = Path(tmp) / (out_path.stem + ".docx")
        markdown_to_docx(markdown, docx_path)
        if _convert_docx_to_pdf(docx_path, out_path):
            return out_path
    _markdown_to_pdf_reportlab(markdown, out_path)
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    raise RuntimeError("PDF export failed: neither the office renderer nor the PDF fallback produced a valid file.")
def _markdown_to_pdf_reportlab(markdown: str, out_path: Path) -> None:
    """Dependency-light PDF fallback using the same block parser as DOCX."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Spacer,
        Table, TableStyle, PageBreak,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("RPBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10.2, leading=13.5, spaceAfter=7)
    h1 = ParagraphStyle("RPH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=21, leading=25, spaceAfter=8, textColor=colors.HexColor("#202020"))
    h2 = ParagraphStyle("RPH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=17, spaceBefore=10, spaceAfter=5)
    h3 = ParagraphStyle("RPH3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=11.2, leading=14, spaceBefore=7, spaceAfter=4)
    meta = ParagraphStyle("RPMeta", parent=body, fontName="Helvetica-Oblique", fontSize=8.5, textColor=colors.HexColor("#666666"), spaceAfter=12)
    quote = ParagraphStyle("RPQuote", parent=body, leftIndent=18, textColor=colors.HexColor("#555E58"), fontName="Times-Italic")
    bullet = ParagraphStyle("RPBullet", parent=body, leftIndent=14, firstLineIndent=-8, spaceAfter=3)
    def inline(text: str) -> str:
        escaped = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        escaped = re.sub(r"\\*\\*(.+?)\\*\\*", r"<b>\\1</b>", escaped)
        escaped = re.sub(r"\\*([^*]+?)\\*", r"<i>\\1</i>", escaped)
        return escaped
    class Doc(BaseDocTemplate):
        def __init__(self, filename):
            super().__init__(filename, pagesize=LETTER, leftMargin=.78*inch, rightMargin=.78*inch, topMargin=.72*inch, bottomMargin=.7*inch)
            frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id="normal")
            self.addPageTemplates([PageTemplate(id="main",frames=frame,onPage=self._header_footer)])
        def _header_footer(self, canvas, doc):
            canvas.saveState(); canvas.setFont("Helvetica",7.5); canvas.setFillColor(colors.HexColor("#777777"))
            canvas.drawString(self.leftMargin, LETTER[1]-0.42*inch, "Research report")
            canvas.drawCentredString(LETTER[0]/2, 0.38*inch, f"{doc.page}")
            canvas.restoreState()
    doc=Doc(str(out_path)); story=[]
    blocks=parse_blocks(markdown)
    for b in blocks:
        if b.kind=="h1": story.append(Paragraph(inline(b.text),h1))
        elif b.kind=="h2": story.append(Paragraph(inline(b.text),h2))
        elif b.kind=="h3": story.append(Paragraph(inline(b.text),h3))
        elif b.kind=="meta": story.append(Paragraph(inline(b.text),meta))
        elif b.kind=="hr": story += [Spacer(1,4)]
        elif b.kind=="bullet": story.append(Paragraph("• " + inline(b.text),bullet))
        elif b.kind=="quote": story.append(Paragraph("“"+inline(b.text)+"”",quote))
        elif b.kind=="para": story.append(Paragraph(inline(b.text),body))
        elif b.kind=="table" and b.rows:
            rows=[[Paragraph(inline(c), body) for c in row] for row in b.rows]
            if rows:
                tbl=Table(rows, repeatRows=1, hAlign="LEFT", colWidths=None)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#333333")),
                    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                    ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#B8B8B8")),
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5),
                    ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
                ]))
                story += [Spacer(1,3), tbl, Spacer(1,5)]
    doc.build(story)
def _latex_escape(text: str) -> str:
    replacements = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        'α': r'$\alpha$', 'β': r'$\beta$', 'γ': r'$\gamma$', 'δ': r'$\delta$',
        'Δ': r'$\Delta$', 'θ': r'$\theta$', 'λ': r'$\lambda$', 'μ': r'$\mu$',
        'π': r'$\pi$', 'σ': r'$\sigma$', 'Σ': r'$\Sigma$', 'φ': r'$\phi$',
        'ω': r'$\omega$', '∞': r'$\infty$', '≤': r'$\leq$', '≥': r'$\geq$',
        '±': r'$\pm$', '×': r'$\times$', '→': r'$\rightarrow$', '←': r'$\leftarrow$',
        '↔': r'$\leftrightarrow$', '∈': r'$\in$', '∑': r'$\sum$', '√': r'$\sqrt{}$',
        '·': r'\textperiodcentered{}', '–': r'--', '—': r'---', '“': '``', '”': "''", '’': "'", '‘': "`",
    }
    return ''.join(replacements.get(ch, ch) for ch in str(text))
def markdown_to_latex(markdown: str) -> str:
    """Export the report as a self-contained, conservative LaTeX document."""
    lines = str(markdown or '').replace('\r\n', '\n').split('\n')
    out = [
        r'\documentclass[11pt]{article}',
        r'\usepackage[margin=1in]{geometry}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{booktabs}',
        r'\usepackage{hyperref}',
        r'\usepackage{enumitem}',
        r'\usepackage{microtype}',
        r'\begin{document}',
    ]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('# '): out += [r'\section*{' + _latex_escape(line[2:]) + '}']; i += 1; continue
        if line.startswith('## '): out += [r'\subsection*{' + _latex_escape(line[3:]) + '}']; i += 1; continue
        if line.startswith('### '): out += [r'\subsubsection*{' + _latex_escape(line[4:]) + '}']; i += 1; continue
        if line.startswith('- '):
            out.append(r'\begin{itemize}')
            while i < len(lines) and lines[i].strip().startswith('- '):
                out.append(r'\item ' + _latex_escape(lines[i].strip()[2:])); i += 1
            out.append(r'\end{itemize}'); continue
        if line.startswith('> '): out += [r'\begin{quote}', _latex_escape(line[2:]), r'\end{quote}']; i += 1; continue
        if line == '---': out.append(r'\hrule'); i += 1; continue
        if line.startswith('|'):
            table = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(set(c) <= set('-: ') for c in row): table.append(row)
                i += 1
            if table:
                cols = max(len(r) for r in table)
                out += [r'\begin{tabular}{' + 'l' * cols + '}', r'\toprule']
                for n, row in enumerate(table):
                    cells = [_latex_escape(c) for c in row] + [''] * (cols - len(row))
                    out.append(' & '.join(cells) + r' \\')
                    if n == 0: out.append(r'\midrule')
                out += [r'\bottomrule', r'\end{tabular}', r'\par\medskip']; continue
        if line.startswith('```'):
            code = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i += 1
            i += 1
            out += [r'\begin{verbatim}', *code, r'\end{verbatim}']; continue
        text = _latex_escape(line)
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        out += [text, r'\par']
        i += 1
    out.append(r'\end{document}')
    return '\n'.join(out) + '\n'
