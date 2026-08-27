from pathlib import Path

from ai_rag_demo.loader import load_and_split, split_text


class TestSplitText:
    def test_splits_long_text_into_overlapping_chunks(self) -> None:
        text = "sentence one. " * 200
        chunks = split_text(text, source="test", chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)

    def test_short_text_stays_in_one_chunk(self) -> None:
        chunks = split_text("just a short sentence.", source="test")
        assert len(chunks) == 1
        assert chunks[0].page_content == "just a short sentence."


class TestLoadAndSplit:
    def test_reads_a_real_file_from_disk(self, tmp_path: Path) -> None:
        document = tmp_path / "doc.txt"
        document.write_text("Account balance is $43.00 as of the last billing cycle.")
        chunks = load_and_split(document)
        assert len(chunks) == 1
        assert "43.00" in chunks[0].page_content
        assert chunks[0].metadata["source"] == str(document)
