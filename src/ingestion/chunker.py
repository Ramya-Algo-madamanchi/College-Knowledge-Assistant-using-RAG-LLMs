import os
import re

import pdfplumber
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "Raw_Data"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db"
)


# ============================================================
# 1. NORMAL PDF LOADER
# ============================================================

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "document_type": "normal",
                "source": pdf_path,
                "page": page_number
            }
        })

    return documents


# ============================================================
# 2. CALENDAR LOADER
# ============================================================

def load_calendar_pdf(pdf_path):
    """
    Extract academic calendar using pdfplumber layout=True.

    Each month becomes a separate document so that
    events retain the correct month and year.
    """

    calendar_documents = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            text = page.extract_text(
                layout=True
            )

            if not text:
                continue

            month_pattern = re.compile(
                r"(?m)^\s*"
                r"(January|February|March|April|May|June|"
                r"July|August|September|October|November|December)"
                r"\s+(\d{4})\s+Date Description"
            )

            matches = list(
                month_pattern.finditer(text)
            )

            if not matches:

                print(
                    f"WARNING: No calendar months found "
                    f"in {os.path.basename(pdf_path)}, "
                    f"page {page_number}"
                )

                continue

            for index, match in enumerate(matches):

                month = match.group(1)
                year = match.group(2)

                start = match.start()

                if index + 1 < len(matches):

                    end = matches[
                        index + 1
                    ].start()

                else:

                    end = len(text)

                month_text = text[
                    start:end
                ].strip()

                heading_pattern = (
                    r"^\s*"
                    + re.escape(month)
                    + r"\s+"
                    + re.escape(year)
                    + r"\s+Date Description"
                )

                month_text = re.sub(
                    heading_pattern,
                    "",
                    month_text,
                    count=1
                ).strip()

                if not month_text:
                    continue

                document = (
                    "Academic Calendar\n"
                    "Academic Year: 2026-27\n"
                    f"Month: {month} {year}\n\n"
                    f"{month_text}"
                )

                metadata = {
                    "document_type": "calendar",
                    "source": pdf_path,
                    "page": page_number,
                    "month": month,
                    "year": year
                }

                calendar_documents.append({
                    "text": document,
                    "metadata": metadata
                })

                print(
                    f"Loaded calendar: "
                    f"{os.path.basename(pdf_path)} "
                    f"-> {month} {year}"
                )

    return calendar_documents


# ============================================================
# 3. TEXT CHUNKER
# ============================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for item in documents:

        text = item["text"]
        metadata = item["metadata"]

        split_texts = splitter.split_text(
            text
        )

        for chunk in split_texts:

            chunk_metadata = metadata.copy()

            chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

    return chunks


# ============================================================
# 4. TIMETABLE PARSER
# ============================================================

def parse_timetable(pdf_path):

    timetable_documents = []

    print(
        f"\nParsing timetable: "
        f"{os.path.basename(pdf_path)}"
    )

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            tables = page.extract_tables()

            if not tables:
                continue

            for table in tables:

                if not table:
                    continue

                # ------------------------------------------------
                # Carry course information across rows
                # ------------------------------------------------

                current_course_code = ""
                current_course = ""

                for row in table:

                    if not row:
                        continue

                    # ------------------------------------------------
                    # Clean cells
                    # ------------------------------------------------

                    cells = []

                    for cell in row:

                        if cell is None:
                            cells.append("")
                        else:
                            cells.append(
                                str(cell).strip()
                            )

                    # Remove completely empty cells
                    cells = [
                        cell
                        for cell in cells
                        if cell
                    ]

                    if not cells:
                        continue

                    row_text = " ".join(
                        cells
                    ).strip()

                    if not row_text:
                        continue

                    # =================================================
                    # COURSE CODE
                    # =================================================

                    course_match = re.search(
                        r"\b[A-Z]{2,6}\d{4}\b",
                        row_text
                    )

                    course_code = ""

                    if course_match:

                        course_code = (
                            course_match.group(0)
                        )

                    # =================================================
                    # DATE
                    # =================================================

                    date_match = re.search(
                        r"\b\d{2}-\d{2}-\d{4}\b",
                        row_text
                    )

                    exam_date = ""

                    if date_match:

                        exam_date = (
                            date_match.group(0)
                        )

                    # =================================================
                    # SESSION
                    # =================================================

                    session = ""

                    if re.search(
                        r"\bAN\b",
                        row_text
                    ):

                        session = "AN"

                    elif re.search(
                        r"\bFN\b",
                        row_text
                    ):

                        session = "FN"

                    # =================================================
                    # STUDENT-SPECIFIC TIMETABLE
                    #
                    # These rows contain more information and require
                    # at least 5 cells.
                    # =================================================

                    registration_match = re.search(
                        r"\b\d{10}\b",
                        row_text
                    )

                    if (
                        registration_match
                        and len(cells) >= 5
                    ):

                        registration_no = (
                            registration_match.group(0)
                        )

                        # ---------------------------------------------
                        # Course code
                        # ---------------------------------------------

                        if course_code:

                            current_course_code = (
                                course_code
                            )

                        # ---------------------------------------------
                        # Course name
                        # ---------------------------------------------

                        course = ""

                        for cell in cells:

                            if not cell:
                                continue

                            if re.fullmatch(
                                r"\d{10}",
                                cell
                            ):
                                continue

                            if re.fullmatch(
                                r"[A-Z]{2,6}\d{4}",
                                cell
                            ):
                                continue

                            if re.fullmatch(
                                r"\d{2}-\d{2}-\d{4}",
                                cell
                            ):
                                continue

                            if cell in [
                                "AN",
                                "FN"
                            ]:
                                continue

                            if len(cell) > 2:

                                course = cell
                                break

                        if course:

                            current_course = course

                        # ---------------------------------------------
                        # Only create record when date exists
                        # ---------------------------------------------

                        if (
                            exam_date
                            and current_course_code
                        ):

                            text = (
                                "Exam Type: "
                                "Student Timetable\n"
                                f"Registration Number: "
                                f"{registration_no}\n"
                                f"Course Code: "
                                f"{current_course_code}\n"
                                f"Course: "
                                f"{current_course}\n"
                                f"Exam Date: "
                                f"{exam_date}\n"
                                f"Session: "
                                f"{session}\n"
                                f"Source Page: "
                                f"{page_number}"
                            )

                            metadata = {
                                "document_type":
                                    "timetable",

                                "timetable_type":
                                    "student_timetable",

                                "registration_no":
                                    registration_no,

                                "course_code":
                                    current_course_code,

                                "course":
                                    current_course,

                                "exam_date":
                                    exam_date,

                                "session":
                                    session,

                                "page":
                                    page_number,

                                "source":
                                    pdf_path
                            }

                            timetable_documents.append({
                                "text": text,
                                "metadata": metadata
                            })

                        # This row has been handled.
                        continue

                    # =================================================
                    # GENERAL TIMETABLE
                    #
                    # General rows require at least 4 cells.
                    #
                    # Important:
                    # Some rows contain the course code/name.
                    # Following rows may contain ONLY date/session.
                    # We therefore carry current_course_code/course.
                    # =================================================

                    if len(cells) >= 4:

                        # ------------------------------------------------
                        # Update current course if this row contains one
                        # ------------------------------------------------

                        if course_code:

                            current_course_code = (
                                course_code
                            )

                            # Find course name from cells
                            course_candidates = []

                            for cell in cells:

                                if not cell:
                                    continue

                                if re.fullmatch(
                                    r"[A-Z]{2,6}\d{4}",
                                    cell
                                ):
                                    continue

                                if re.fullmatch(
                                    r"\d{2}-\d{2}-\d{4}",
                                    cell
                                ):
                                    continue

                                if cell in [
                                    "AN",
                                    "FN"
                                ]:
                                    continue

                                if len(cell) > 2:

                                    course_candidates.append(
                                        cell
                                    )

                            if course_candidates:

                                current_course = (
                                    course_candidates[0]
                                )

                        # ------------------------------------------------
                        # Create general timetable record
                        # ------------------------------------------------

                        if (
                            exam_date
                            and current_course_code
                        ):

                            text = (
                                "Exam Type: "
                                "General Timetable\n"
                                f"Course Code: "
                                f"{current_course_code}\n"
                                f"Course: "
                                f"{current_course}\n"
                                f"Exam Date: "
                                f"{exam_date}\n"
                                f"Session: "
                                f"{session}\n"
                                f"Source Page: "
                                f"{page_number}"
                            )

                            metadata = {
                                "document_type":
                                    "timetable",

                                "timetable_type":
                                    "general_timetable",

                                "course_code":
                                    current_course_code,

                                "course":
                                    current_course,

                                "exam_date":
                                    exam_date,

                                "session":
                                    session,

                                "page":
                                    page_number,

                                "source":
                                    pdf_path
                            }

                            timetable_documents.append({
                                "text": text,
                                "metadata": metadata
                            })

    print(
        f"Timetable records extracted: "
        f"{len(timetable_documents)}"
    )

    return timetable_documents


# ============================================================
# 5. LOAD ALL DOCUMENTS
# ============================================================

all_documents = []

for filename in sorted(
    os.listdir(DATA_DIR)
):

    if not filename.lower().endswith(
        ".pdf"
    ):
        continue

    pdf_path = os.path.join(
        DATA_DIR,
        filename
    )

    # --------------------------------------------------------
    # Calendar PDFs
    # --------------------------------------------------------

    if filename in [
        "academic_calendar_2026_27_odd_seniors.pdf",
        "academic_calendar_2026_27_even_seniors.pdf"
    ]:

        calendar_documents = (
            load_calendar_pdf(pdf_path)
        )

        all_documents.extend(
            calendar_documents
        )

        continue

    # --------------------------------------------------------
    # Large timetable
    # --------------------------------------------------------

    if filename == (
        "cse_sessional_exam_timetable_april_2026_sem_iv_vi.pdf.pdf"
    ):

        timetable_documents = (
            parse_timetable(pdf_path)
        )

        all_documents.extend(
            timetable_documents
        )

        continue

    # --------------------------------------------------------
    # Normal PDFs
    # --------------------------------------------------------

    documents = load_pdf(
        pdf_path
    )

    print(
        f"Loaded: {filename} | "
        f"Pages: {len(documents)}"
    )

    all_documents.extend(
        documents
    )


# ============================================================
# 6. SEPARATE DOCUMENT TYPES
# ============================================================

normal_documents = [
    item
    for item in all_documents
    if item["metadata"].get(
        "document_type"
    ) == "normal"
]

calendar_documents = [
    item
    for item in all_documents
    if item["metadata"].get(
        "document_type"
    ) == "calendar"
]

timetable_documents = [
    item
    for item in all_documents
    if item["metadata"].get(
        "document_type"
    ) == "timetable"
]


# ============================================================
# 7. CHUNK NORMAL DOCUMENTS
# ============================================================

normal_chunks = create_chunks(
    normal_documents
)


# ============================================================
# 8. FINAL DOCUMENT LIST
# ============================================================

final_documents = (
    normal_chunks
    + calendar_documents
    + timetable_documents
)


print("\n========================================")
print("NORMAL DOCUMENTS")
print("========================================")

print(
    f"Normal pages: "
    f"{len(normal_documents)}"
)

print(
    f"Normal chunks: "
    f"{len(normal_chunks)}"
)


print("\n========================================")
print("CALENDAR")
print("========================================")

print(
    f"Calendar documents: "
    f"{len(calendar_documents)}"
)


print("\n========================================")
print("TIMETABLE")
print("========================================")

print(
    f"Timetable records: "
    f"{len(timetable_documents)}"
)


print("\n========================================")
print("FINAL DOCUMENT COUNT")
print("========================================")

print(
    f"Total documents to store: "
    f"{len(final_documents)}"
)


# ============================================================
# 9. PREPARE TEXT + METADATA
# ============================================================

texts = [
    item["text"]
    for item in final_documents
]

metadatas = [
    item["metadata"]
    for item in final_documents
]


# ============================================================
# 10. LOAD MINILM EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 11. CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)


# ============================================================
# 12. CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR
)

try:

    client.delete_collection(
        name="college_knowledge"
    )

    print(
        "Old collection deleted."
    )

except Exception:

    print(
        "No old collection found."
    )


collection = client.create_collection(
    name="college_knowledge"
)


# ============================================================
# 13. STORE DOCUMENTS
# ============================================================

BATCH_SIZE = 5000

for start in range(
    0,
    len(texts),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(texts)
    )

    collection.add(
        ids=[
            str(i)
            for i in range(
                start,
                end
            )
        ],

        documents=texts[
            start:end
        ],

        metadatas=metadatas[
            start:end
        ],

        embeddings=embeddings[
            start:end
        ].tolist()
    )

    print(
        f"Stored documents: "
        f"{start} - {end}"
    )


# ============================================================
# 14. FINAL CHROMADB CHECK
# ============================================================

print("\n========================================")
print("CHROMADB COMPLETE")
print("========================================")

print(
    "Documents stored:",
    collection.count()
)


# ============================================================
# 15. PSYC2071 CHECK
# ============================================================

print("\n========================================")
print("PSYC2071 CHROMADB CHECK")
print("========================================")

results = collection.get(
    where={
        "course_code": "PSYC2071"
    }
)

print(
    "PSYC2071 documents in ChromaDB:",
    len(results["ids"])
)

if results["ids"]:

    print("\nMetadata:")
    print(
        results["metadatas"][0]
    )

    print("\nDocument:")
    print(
        results["documents"][0]
    )


# ============================================================
# 16. CALENDAR CHECK
# ============================================================

print("\n========================================")
print("CALENDAR CHECK")
print("========================================")

calendar_results = collection.get(
    where={
        "document_type": "calendar"
    }
)

print(
    "Calendar documents:",
    len(calendar_results["ids"])
)


for document, metadata in zip(
    calendar_results["documents"][:10],
    calendar_results["metadatas"][:10]
):

    print("\n----------------------------------------")

    print("Metadata:")
    print(metadata)

    print("\nDocument:")
    print(document[:500])