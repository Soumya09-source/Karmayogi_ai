import os

from docx import Document
from pypdf import PdfReader


def extract_text(file_data, filename: str) -> str:
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".txt":
        return file_data.read().decode("utf-8")

    if extension == ".pdf":
        reader = PdfReader(file_data)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    if extension == ".docx":
        document = Document(file_data)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

    raise ValueError(
        "Unsupported file type. Supported types: TXT, PDF, DOCX"
    )
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:

    text = text.strip()

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Normalize whitespace while preserving paragraph boundaries.
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    # Convert paragraphs into complete sentences.
    sentences = []

    for paragraph in paragraphs:
        current = ""

        for char in paragraph:
            current += char

            if char in ".?!":
                sentence = current.strip()

                if sentence:
                    sentences.append(sentence)

                current = ""

        # Keep any remaining text as a complete unit.
        if current.strip():
            sentences.append(current.strip())

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:

        # If adding this sentence stays within the target size,
        # keep it in the current chunk.
        if (
            current_chunk
            and current_length + len(sentence) + 1 > chunk_size
        ):
            chunks.append(" ".join(current_chunk))

            # Build overlap using complete previous sentences.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(current_chunk):
                if overlap_length + len(previous) + 1 > overlap:
                    break

                overlap_sentences.insert(0, previous)
                overlap_length += len(previous) + 1

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += len(sentence) + 1

    # Add the final chunk.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Merge a very small final chunk with the previous chunk.
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size * 0.2:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks