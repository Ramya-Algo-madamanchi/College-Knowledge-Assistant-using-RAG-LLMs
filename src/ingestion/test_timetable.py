import pdfplumber
from pathlib import Path


def parse_timetable(pdf_path):

    records = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            tables = page.extract_tables()

            for table in tables:

                if not table or len(table) < 2:
                    continue

                header = " ".join(
                    str(cell)
                    for cell in table[1]
                    if cell
                ).lower()

                # ==================================================
                # GENERAL TIMETABLE
                # ==================================================

                if (
                    "course code" in header
                    and "title of course" in header
                    and "redg no" not in header
                ):

                    current_course_code = None
                    current_course = None

                    for row in table[2:]:

                        if not row or len(row) < 4:
                            continue

                        course_code = row[0]
                        course_name = row[1]
                        exam_date = row[2]
                        session = row[3]

                        # New course
                        if course_code and course_name:

                            current_course_code = course_code.strip()
                            current_course = course_name.strip()

                        # Date for current course
                        if (
                            exam_date
                            and session
                            and current_course_code
                            and current_course
                        ):

                            records.append({
                                "type": "general_timetable",
                                "course_code": current_course_code,
                                "course": current_course,
                                "exam_date": exam_date.strip(),
                                "session": session.strip(),
                                "page": page_number
                            })

                # ==================================================
                # STUDENT-SPECIFIC TIMETABLE
                # ==================================================

                elif (
                    "redg no" in header
                    and "course code" in header
                ):

                    for row in table[2:]:

                        if not row or len(row) < 5:
                            continue

                        registration_no = row[0]
                        course_code = row[1]
                        course_name = row[2]
                        exam_date = row[3]
                        session = row[4]

                        if not all([
                            registration_no,
                            course_code,
                            course_name,
                            exam_date,
                            session
                        ]):
                            continue

                        records.append({
                            "type": "student_timetable",
                            "registration_no": registration_no.strip(),
                            "course_code": course_code.strip(),
                            "course": course_name.strip(),
                            "exam_date": exam_date.strip(),
                            "session": session.strip(),
                            "page": page_number
                        })

    return records


# ==========================================================
# CONVERT RECORDS TO RAG TEXT
# ==========================================================

def create_rag_documents(records):

    documents = []

    for record in records:

        if record["type"] == "general_timetable":

            text = (
                f"Exam Type: General Timetable\n"
                f"Course Code: {record['course_code']}\n"
                f"Course: {record['course']}\n"
                f"Exam Date: {record['exam_date']}\n"
                f"Session: {record['session']}\n"
                f"Source Page: {record['page']}"
            )

        else:

            text = (
                f"Exam Type: Student-Specific Timetable\n"
                f"Registration Number: "
                f"{record['registration_no']}\n"
                f"Course Code: {record['course_code']}\n"
                f"Course: {record['course']}\n"
                f"Exam Date: {record['exam_date']}\n"
                f"Session: {record['session']}\n"
                f"Source Page: {record['page']}"
            )

        documents.append(text)

    return documents


# ==========================================================
# TEST
# ==========================================================

pdf_path = Path(
    "Data/Raw_Data/"
    "cse_sessional_exam_timetable_april_2026_sem_iv_vi.pdf.pdf"
)

records = parse_timetable(pdf_path)

rag_documents = create_rag_documents(records)


print("\n========== TOTAL RECORDS ==========")
print(len(records))


print("\n========== PSYC2071 RAG DOCUMENT ==========\n")

for record, document in zip(records, rag_documents):

    if record["course_code"] == "PSYC2071":

        print(document)


print("\n========== FIRST 3 RAG DOCUMENTS ==========\n")

for document in rag_documents[:3]:

    print(document)
    print("\n-----------------------------\n")