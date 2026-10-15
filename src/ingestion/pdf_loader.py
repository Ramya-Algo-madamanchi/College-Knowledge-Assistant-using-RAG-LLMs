from pypdf import PdfReader
from pathlib import Path


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        documents.append({
            "text": text,
            "source": str(pdf_path),
            "page": page_number
        })

    return documents


data_folder = Path("Data/Raw_Data")

all_documents = []

for pdf_file in data_folder.glob("*.pdf"):
    documents = load_pdf(pdf_file)
    all_documents.extend(documents)

    print(f"Loaded: {pdf_file.name} | Pages: {len(documents)}")


print("\nTotal pages loaded:", len(all_documents))