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

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks