import re


def split_markdown_by_sections(
    markdown_text: str,
) -> list[dict[str, str]]:
    """
    Split a Markdown document by its level-2 headings (##).
    Keep the document title and metadata in every chunk.
    """
    text = markdown_text.strip()

    if not text:
        return []

    headings = list(
        re.finditer(r"(?m)^##\s+.+$", text)
    )

    if not headings:
        return [
            {
                "section_title": "Full Document",
                "text": text,
            }
        ]

    preamble = text[:headings[0].start()].strip()
    chunks = []

    for index, heading in enumerate(headings):
        start = heading.start()

        if index + 1 < len(headings):
            end = headings[index + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()
        section_title = heading.group().lstrip("#").strip()

        if preamble:
            chunk_text = f"{preamble}\n\n{section_text}"
        else:
            chunk_text = section_text

        chunks.append(
            {
                "section_title": section_title,
                "text": chunk_text,
            }
        )

    return chunks
    
    
    
if __name__ == "__main__":
    from pathlib import Path

    document_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "documents"
        / "leave_policy.md"
    )

    markdown_text = document_path.read_text(
        encoding="utf-8"
    )

    chunks = split_markdown_by_sections(markdown_text)

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Section: {chunk['section_title']}")
        print(chunk["text"])
