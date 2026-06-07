"""
Deep tests for parseDocument.py — LlamaCloud SDK v2 integration.

API response structure (verified from SDK introspection):
    result.text         → Text object with .pages: List[TextPage]
    TextPage            → .page_number (int), .text (str)
    result.items        → Items object with .pages: List[ItemsPage]

These tests verify:
- parsing.parse() is called with upload_file= (no separate upload step)
- Correct arguments: tier, version, expand
- Page normalisation from TextPage objects
- Items grouped correctly per page
- Edge cases: empty text, None items, missing pages
"""
from unittest.mock import MagicMock, patch, mock_open
import pytest

from topictrace.rag.documentIngestion.parseDocument import parse_document, get_all_pages_text



# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_text_page(page_number: int, text: str) -> MagicMock:
    """Build a fake TextPage with .page_number and .text."""
    page = MagicMock()
    page.page_number = page_number
    page.text = text
    return page

def _make_items_page(page_number: int) -> MagicMock:
    """Build a fake ItemsPage."""
    page = MagicMock()
    page.page_number = page_number
    return page


def _make_api_result(text_pages: list, items_pages: list) -> MagicMock:
    """Build a fake LlamaCloud parse result with nested Text and Items objects."""
    result = MagicMock()
    result.text = MagicMock()
    result.text.pages = text_pages
    result.items = MagicMock()
    result.items.pages = items_pages
    return result


def _make_client(api_result: MagicMock) -> MagicMock:
    """Build a fake LlamaCloud client."""
    client = MagicMock()
    client.parsing.parse.return_value = api_result
    return client


# ── Tests: parse_document() ───────────────────────────────────────────────────

def test_parse_document_uses_single_call_with_upload_file():
    """The API requires a single parsing.parse(upload_file=...) — no separate upload step."""
    api_result = _make_api_result([_make_text_page(1, "content")], [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"pdf bytes")):
        parse_document("some/file.pdf")

    client.files.create.assert_not_called()
    client.parsing.parse.assert_called_once()


def test_parse_document_passes_upload_file_kwarg():
    """parsing.parse() must receive upload_file= — not file_id= from a separate upload."""
    api_result = _make_api_result([_make_text_page(1, "content")], [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        parse_document("my_doc.pdf")

    parse_call_kwargs = client.parsing.parse.call_args[1]
    assert "upload_file" in parse_call_kwargs, "parse() must use upload_file= kwarg"
    assert "file_id" not in parse_call_kwargs, "file_id= must not be used — no separate upload step"


def test_parse_document_requests_text_and_items_expand():
    """Must expand=['text', 'items'] — not markdown."""
    api_result = _make_api_result([_make_text_page(1, "content")], [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        parse_document("my_doc.pdf")

    parse_call_kwargs = client.parsing.parse.call_args[1]
    expand = parse_call_kwargs["expand"]

    assert "text" in expand, "expand must include 'text'"
    assert "items" in expand, "expand must include 'items' for structured JSON"
    assert "markdown" not in expand, "markdown expand must not be used"


def test_parse_document_uses_cost_effective_tier():
    """Default tier must be 'cost_effective' to control API spend."""
    api_result = _make_api_result([_make_text_page(1, "content")], [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        parse_document("my_doc.pdf")

    assert client.parsing.parse.call_args[1]["tier"] == "cost_effective"


def test_parse_document_returns_correct_number_of_pages():
    """One TextPage object in result → one page dict in output."""
    text_pages = [_make_text_page(1, "p1"), _make_text_page(2, "p2"), _make_text_page(3, "p3")]
    api_result = _make_api_result(text_pages, [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    assert len(pages) == 3


def test_parse_document_page_numbers_come_from_text_page_object():
    """page number in output must come from TextPage.page_number, not enumerate()."""
    text_pages = [_make_text_page(1, "A"), _make_text_page(2, "B")]
    api_result = _make_api_result(text_pages, [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2


def test_parse_document_strips_whitespace_from_page_text():
    """Text from each TextPage must be stripped."""
    text_pages = [_make_text_page(1, "  some content  ")]
    api_result = _make_api_result(text_pages, [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    assert pages[0]["text"] == "some content"


def test_parse_document_attaches_items_page_to_correct_page():
    """ItemsPage for page 2 must appear in pages[1]['items'], not pages[0]."""
    text_pages = [_make_text_page(1, "p1"), _make_text_page(2, "p2")]
    items_page_2 = _make_items_page(page_number=2)
    api_result = _make_api_result(text_pages, [items_page_2])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    assert pages[0]["items"] == [], "Page 1 must have no items"
    assert pages[1]["items"] is items_page_2, "Page 2 must have its ItemsPage"


def test_parse_document_page_with_no_items_gets_empty_list():
    """Pages with no matching ItemsPage must get items=[] not None."""
    text_pages = [_make_text_page(1, "p1")]
    api_result = _make_api_result(text_pages, [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    assert pages[0]["items"] == []


def test_parse_document_handles_no_text_in_response():
    """If result.text is None or has no pages, output must be empty list — not crash."""
    api_result = MagicMock()
    api_result.text = None
    api_result.items = None
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    assert pages == []


def test_parse_document_each_page_dict_has_required_keys():
    """Every returned page dict MUST have: page, text, items."""
    text_pages = [_make_text_page(1, "hello"), _make_text_page(2, "world")]
    api_result = _make_api_result(text_pages, [])
    client = _make_client(api_result)

    with patch("topictrace.rag.documentIngestion.parseDocument._build_client", return_value=client), \
         patch("builtins.open", mock_open(read_data=b"bytes")):
        pages = parse_document("doc.pdf")

    for i, page in enumerate(pages):
        assert "page" in page, f"Page {i} missing 'page' key"
        assert "text" in page, f"Page {i} missing 'text' key"
        assert "items" in page, f"Page {i} missing 'items' key"


# ── Tests: get_all_pages_text() ───────────────────────────────────────────────

def test_get_all_pages_text_formats_with_page_headers():
    pages = [
        {"page": 1, "text": "Alice works here.", "items": []},
        {"page": 2, "text": "Bob manages Alice.", "items": []},
    ]
    result = get_all_pages_text(pages)
    assert "--- Page 1 ---" in result
    assert "--- Page 2 ---" in result


def test_get_all_pages_text_preserves_all_page_content():
    pages = [
        {"page": 1, "text": "CONTENT_PAGE_ONE", "items": []},
        {"page": 2, "text": "CONTENT_PAGE_TWO", "items": []},
    ]
    result = get_all_pages_text(pages)
    assert "CONTENT_PAGE_ONE" in result
    assert "CONTENT_PAGE_TWO" in result


def test_get_all_pages_text_separates_pages_with_blank_line():
    pages = [{"page": 1, "text": "p1", "items": []}, {"page": 2, "text": "p2", "items": []}]
    assert "\n\n" in get_all_pages_text(pages)


def test_get_all_pages_text_on_empty_input_returns_empty_string():
    assert get_all_pages_text([]) == ""


def test_get_all_pages_text_handles_page_with_empty_text():
    pages = [{"page": 1, "text": "", "items": []}]
    result = get_all_pages_text(pages)
    assert "--- Page 1 ---" in result
