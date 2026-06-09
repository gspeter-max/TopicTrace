"""
Tests for topictrace.rag.documentIngestion.chunking.recursive.
"""

from topictrace.rag.documentIngestion.chunking import chunk_document, count_tokens

SIMPLE_TEXT = "Hello world. " * 100
MULTI_PARAGRAPH_TEXT = (
    "This is paragraph one. " * 30
    + "\n\n"
    + "This is paragraph two. " * 30
    + "\n\n"
    + "This is paragraph three. " * 30
    + "\n\n"
    + "This is paragraph four. " * 30
    + "\n\n"
    + "This is paragraph five. " * 30
)


class TestCountTokens:
    def test_returns_integer(self):
        result = count_tokens("Hello world")
        assert isinstance(result, int)

    def test_empty_string(self):
        result = count_tokens("")
        assert result >= 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hello")
        long = count_tokens("Hello world this is a longer piece of text")
        assert long > short

    def test_token_count_is_reasonable(self):
        text = "word " * 100
        tokens = count_tokens(text)
        assert 80 <= tokens <= 200


class TestChunkDocument:
    def test_returns_list(self):
        result = chunk_document(SIMPLE_TEXT, document_id="test.pdf")
        assert isinstance(result, list)

    def test_each_item_is_dict(self):
        result = chunk_document(SIMPLE_TEXT, document_id="test.pdf")
        for item in result:
            assert isinstance(item, dict)

    def test_required_keys_present(self):
        result = chunk_document(SIMPLE_TEXT, document_id="test.pdf")
        required_keys = {"chunk_index", "text", "token_count", "document_id"}
        for chunk in result:
            assert required_keys.issubset(chunk.keys()), (
                f"Missing keys: {required_keys - chunk.keys()}"
            )

    def test_chunk_index_sequential(self):
        result = chunk_document(MULTI_PARAGRAPH_TEXT, document_id="test.pdf")
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i

    def test_document_id_propagated(self):
        doc_id = "my_resume.pdf"
        result = chunk_document(SIMPLE_TEXT, document_id=doc_id)
        for chunk in result:
            assert chunk["document_id"] == doc_id

    def test_no_chunk_exceeds_max_tokens(self):
        result = chunk_document(MULTI_PARAGRAPH_TEXT, document_id="test.pdf")
        for chunk in result:
            assert chunk["token_count"] <= 512, (
                f"Chunk {chunk['chunk_index']} has {chunk['token_count']} tokens (max 512)"
            )

    def test_token_count_matches_text(self):
        result = chunk_document(MULTI_PARAGRAPH_TEXT, document_id="test.pdf")
        for chunk in result:
            actual = count_tokens(chunk["text"])
            assert chunk["token_count"] == actual, (
                f"Chunk {chunk['chunk_index']}: stored {chunk['token_count']}, actual {actual}"
            )

    def test_multiple_chunks_produced_for_long_text(self):
        long_text = "This is a test sentence with several words. " * 120
        result = chunk_document(long_text, document_id="test.pdf")
        assert len(result) >= 2, "Long text should produce at least 2 chunks"

    def test_short_text_produces_one_chunk(self):
        short_text = "Short document. Just a few sentences."
        result = chunk_document(short_text, document_id="test.pdf")
        assert len(result) == 1

    def test_custom_chunk_size(self):
        result = chunk_document(
            MULTI_PARAGRAPH_TEXT,
            document_id="test.pdf",
            chunk_size=256,
            chunk_overlap=50,
        )
        for chunk in result:
            assert chunk["token_count"] <= 256

    def test_text_is_not_empty_string(self):
        result = chunk_document(MULTI_PARAGRAPH_TEXT, document_id="test.pdf")
        for chunk in result:
            assert chunk["text"].strip() != ""

    def test_empty_document_returns_empty_list(self):
        result = chunk_document("", document_id="test.pdf")
        assert result == []
