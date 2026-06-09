"""
Document parsing using the LlamaCloud SDK (llama-cloud >= 2.0).

Uses the single-call parsing.parse() API with upload_file= parameter.
Returns structured per-page data using the Text and Items expand options.

Response structure from the API:
    result.text         → Text object with .pages: List[TextPage]
    result.text_full    → full document as a single string
    TextPage            → .page_number (int), .text (str)
    result.items        → Items object with .pages: List[ItemsPage]
"""

from llama_cloud import LlamaCloud

from topictrace import settings


def _build_client() -> LlamaCloud:
    """Initialise the LlamaCloud client using the API key from config."""
    return LlamaCloud(api_key=settings.LLAMA_PARSE_APIKEY)


def parse_document(file_path: str) -> list[dict]:
    """
    Parse a document via LlamaCloud and return a normalised list of page dicts.

    Each dict in the returned list contains:
        "page"   → 1-based page number (int)
        "text"   → plain text for that page (str)
        "items"  → list of structured item objects for that page (headings, tables, etc.)

    Args:
        file_path: Absolute or relative path to the document (PDF, DOCX, …).

    Returns:
        A list of page dicts, one per page in the document.
    """
    client = _build_client()

    # Single call: upload + parse + poll until complete
    with open(file_path, "rb") as f:
        result = client.parsing.parse(
            upload_file=(file_path, f),
            tier="cost_effective",
            version="latest",
            expand=["text", "items"],
        )

    # result.text is a Text object with .pages: List[TextPage]
    # Each TextPage has .page_number (int) and .text (str)
    pages: list[dict] = []

    if result.text and result.text.pages:
        # Build a lookup: page_number → list of item objects
        items_by_page: dict[int, list] = {}
        if result.items and result.items.pages:
            for items_page in result.items.pages:
                page_num = items_page.page_number
                # ItemsPage holds the structured blocks for that page
                items_by_page[page_num] = items_page

        for text_page in result.text.pages:
            page_num = text_page.page_number
            pages.append(
                {
                    "page": page_num,
                    "text": (text_page.text or "").strip(),
                    "items": items_by_page.get(page_num, []),
                }
            )

    return pages


def get_all_pages_text(pages: list[dict]) -> str:
    """
    Concatenate all page texts from the output of parse_document().

    Args:
        pages: The list returned by parse_document().

    Returns:
        A single string with each page's text separated by a blank line,
        prefixed with a header marker:

            --- Page 1 ---
            <text>

            --- Page 2 ---
            <text>
    """
    page_texts: list[str] = []
    for page in pages:
        page_number = page.get("page", "?")
        text = page.get("text", "").strip()
        page_texts.append(f"--- Page {page_number} ---\n{text}")

    return "\n\n".join(page_texts)
