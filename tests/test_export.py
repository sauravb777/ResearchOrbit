from researchpilot.export import markdown_to_docx, markdown_to_pdf, sources_to_bibtex, _convert_docx_to_pdf
from researchpilot.schemas import Source
_SAMPLE_WITH_UNICODE = (
    "# Research Report: Test\n\n"
    "## Executive Answer\n\n"
    "- A finding with a smart quote \u2018like this\u2019 and an em dash \u2014 here [E1]\n"
    "- **Bold point** with an arrow \u2192 result [E2]\n\n"
    "## References\n\n"
    "- **[S1]** Some Paper. Author \u00b7 2025 \u00b7 Venue\n"
)
def test_markdown_to_docx_produces_a_nonempty_file(tmp_path):
    out = markdown_to_docx(_SAMPLE_WITH_UNICODE, tmp_path / "report.docx")
    assert out.exists()
    assert out.stat().st_size > 0
def test_markdown_to_pdf_has_a_real_fallback_when_office_renderer_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("researchpilot.export._convert_docx_to_pdf", lambda docx, pdf: False)
    out = markdown_to_pdf(_SAMPLE_WITH_UNICODE, tmp_path / "report.pdf")
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes().startswith(b"%PDF")
def test_sources_to_bibtex_generates_valid_entries():
    sources = [
        Source(id="1", title="A Great Paper", authors="Jane Doe, John Smith", year="2025",
               venue="NeurIPS", doi="10.1234/xyz", provider="arxiv"),
        Source(id="2", title="Another Paper", authors="", year="", venue="", provider="arxiv",
               url="https://arxiv.org/abs/1234"),
    ]
    bib = sources_to_bibtex(sources)
    assert "@article{doe2025," in bib
    assert "title = {A Great Paper}" in bib
    assert "@misc{anonn_d," in bib
def test_sources_to_bibtex_disambiguates_duplicate_citekeys():
    sources = [
        Source(id="1", title="First Paper", authors="Jane Doe", year="2025", provider="a"),
        Source(id="2", title="Second Paper", authors="Jane Doe", year="2025", provider="a"),
    ]
    bib = sources_to_bibtex(sources)
    assert "@misc{doe2025," in bib
    assert "@misc{doe2025a," in bib
def test_latex_export_escapes_common_unicode_math_symbols():
    from researchpilot.export import markdown_to_latex
    tex = markdown_to_latex("# Test\n\nα β ≥ → 2026")
    assert r"$\alpha$" in tex and r"$\beta$" in tex and r"$\geq$" in tex and r"$\rightarrow$" in tex
