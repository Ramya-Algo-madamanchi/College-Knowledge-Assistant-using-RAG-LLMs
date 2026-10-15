import re
import math
import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_collection(
    name="college_knowledge"
)


# ==================================================
# 2. LOAD EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 3. LOAD ALL DOCUMENTS
# ==================================================

all_data = collection.get(
    include=["documents", "metadatas"]
)

all_documents = all_data["documents"]
all_metadatas = all_data["metadatas"]

total_documents = len(all_documents)


# ==================================================
# 4. STOP WORDS
# ==================================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "for",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "how",
    "many",
    "much",
    "does",
    "do",
    "required",
    "need",
    "needs",
    "tell",
    "me",
    "can",
    "i",
    "my",
    "with",
    "from",
    "by",
    "be",
    "at",
    "which",
    "when",
    "where",
    "who",
    "about"
}


# ==================================================
# 5. TOKENIZE
# ==================================================

def tokenize(text):

    words = re.findall(
        r"\b[a-zA-Z0-9.%-]+\b",
        text.lower()
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


# ==================================================
# 6. DOCUMENT FREQUENCY
# ==================================================

document_frequency = {}

for document in all_documents:

    unique_words = set(
        tokenize(document)
    )

    for word in unique_words:

        document_frequency[word] = (
            document_frequency.get(word, 0) + 1
        )


# ==================================================
# 7. QUESTION TYPE DETECTION
# ==================================================

def detect_question_type(question):

    question_lower = question.lower()

    calendar_terms = [
        "calendar",
        "semester",
        "working day",
        "last working day",
        "closure",
        "class work",
        "academic year",
        "academic calendar",
        "odd semester",
        "even semester",
        "date",
        "when"
    ]

    timetable_terms = [
        "timetable",
        "exam date",
        "examination date",
        "session",
        "registration number",
        "course code"
    ]

    is_calendar_question = any(
        term in question_lower
        for term in calendar_terms
    )

    is_timetable_question = any(
        term in question_lower
        for term in timetable_terms
    )

    return (
        is_calendar_question,
        is_timetable_question
    )


# ==================================================
# 8. LEXICAL SEARCH
# ==================================================

def lexical_search(question, top_k=20):

    query_words = tokenize(question)

    if not query_words:
        return []

    question_lower = question.lower()

    (
        is_calendar_question,
        is_timetable_question
    ) = detect_question_type(question)

    scored_documents = []

    # --------------------------------------------------
    # Score every document
    # --------------------------------------------------

    for index, document in enumerate(all_documents):

        document_lower = document.lower()

        document_words = tokenize(document)

        if not document_words:
            continue

        score = 0.0

        # --------------------------------------------------
        # Word counts
        # --------------------------------------------------

        word_counts = {}

        for word in document_words:

            word_counts[word] = (
                word_counts.get(word, 0) + 1
            )

        # --------------------------------------------------
        # Exact phrase match
        # --------------------------------------------------

        if question_lower in document_lower:
            score += 20.0

        # --------------------------------------------------
        # Keyword matching with IDF
        # --------------------------------------------------

        for word in query_words:

            if word not in word_counts:
                continue

            df = document_frequency.get(
                word,
                total_documents
            )

            idf = math.log(
                (total_documents + 1)
                / (df + 1)
            ) + 1

            score += idf

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = all_metadatas[index]

        document_type = metadata.get(
            "document_type",
            "normal"
        )

        source = metadata.get(
            "source",
            ""
        ).lower()

        # ==================================================
        # CALENDAR QUESTION BOOST
        # ==================================================

        if is_calendar_question:

            # Prefer actual academic calendar files
            if (
                "academic_calendar" in source
                or "calendar" in source
            ):
                score += 8.0

            # ----------------------------------------------
            # Odd / Even semester matching
            # ----------------------------------------------

            if "odd semester" in question_lower:

                if "odd" in source:
                    score += 8.0

                if "even" in source:
                    score -= 8.0

            elif "even semester" in question_lower:

                if "even" in source:
                    score += 8.0

                if "odd" in source:
                    score -= 8.0

            # ----------------------------------------------
            # Academic year matching
            # ----------------------------------------------

            if "2026-27" in question_lower:

                if (
                    "2026_27" in source
                    or "2026-27" in document_lower
                ):
                    score += 5.0

            # ----------------------------------------------
            # "Last working day" meaning
            # ----------------------------------------------

            if (
                "last working day" in question_lower
                or "last day" in question_lower
            ):

                if "closure of class work" in document_lower:
                    score += 10.0

                if "last date" in document_lower:
                    score += 4.0

            # ----------------------------------------------
            # Calendar event matching
            # ----------------------------------------------

            calendar_content_terms = [
                "closure of class work",
                "sessional",
                "class work",
                "non-instructional",
                "semester break"
            ]

            for term in calendar_content_terms:

                if term in document_lower:
                    score += 1.5

        # ==================================================
        # TIMETABLE QUESTION BOOST
        # ==================================================

        if is_timetable_question:

            if document_type == "timetable":
                score += 12.0

        # --------------------------------------------------
        # Store scored document
        # --------------------------------------------------

        if score > 0:

            scored_documents.append(
                (
                    score,
                    index
                )
            )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored_documents[:top_k]


# ==================================================
# 9. COURSE CODE DETECTION
# ==================================================

def detect_course_code(question):

    match = re.search(
        r"\b(?:[A-Z]{3,5}\d{4}|\d{2}[A-Z]{3,5}\d{4})\b",
        question.upper()
    )

    if match:
        return match.group()

    return None


# ==================================================
# 10. MAIN RETRIEVAL FUNCTION
# ==================================================

def retrieve(question, top_k=5):

    # ==================================================
    # EXACT COURSE CODE RETRIEVAL
    # ==================================================

    course_code = detect_course_code(question)

    if course_code:

        print(
            f"\nExact course code detected: "
            f"{course_code}"
        )

        course_results = collection.get(
            where={
                "course_code": course_code
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = course_results["documents"]
        metadatas = course_results["metadatas"]

        print(
            f"Documents found for "
            f"{course_code}: {len(documents)}"
        )

        if documents:

            return {
                "documents": [
                    documents[:top_k]
                ],
                "metadatas": [
                    metadatas[:top_k]
                ]
            }

        print(
            "No exact course-code metadata found."
        )

        print(
            "Falling back to hybrid retrieval..."
        )

    # ==================================================
    # SEMANTIC SEARCH
    # ==================================================

    question_embedding = embedding_model.encode(
        question
    )

    semantic_results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=20
    )

    semantic_documents = (
        semantic_results["documents"][0]
    )

    semantic_metadatas = (
        semantic_results["metadatas"][0]
    )

    # ==================================================
    # LEXICAL SEARCH
    # ==================================================

    lexical_results = lexical_search(
        question,
        top_k=20
    )

    # ==================================================
    # COMBINE RESULTS
    # ==================================================

    candidates = {}

    # --------------------------------------------------
    # Add semantic results
    # --------------------------------------------------

    for rank, (
        document,
        metadata
    ) in enumerate(
        zip(
            semantic_documents,
            semantic_metadatas
        )
    ):

        key = (
            metadata.get("source", ""),
            metadata.get("page", ""),
            document
        )

        candidates[key] = {
            "document": document,
            "metadata": metadata,
            "semantic_rank": rank,
            "lexical_rank": None,
            "lexical_score": 0.0
        }

    # --------------------------------------------------
    # Add lexical results
    # --------------------------------------------------

    for rank, (
        lexical_score,
        index
    ) in enumerate(lexical_results):

        document = all_documents[index]

        metadata = all_metadatas[index]

        key = (
            metadata.get("source", ""),
            metadata.get("page", ""),
            document
        )

        if key not in candidates:

            candidates[key] = {
                "document": document,
                "metadata": metadata,
                "semantic_rank": None,
                "lexical_rank": rank,
                "lexical_score": lexical_score
            }

        else:

            candidates[key]["lexical_rank"] = rank
            candidates[key]["lexical_score"] = lexical_score

    # ==================================================
    # HYBRID RANKING
    # ==================================================

    ranked_results = []

    for item in candidates.values():

        score = 0.0

        # --------------------------------------------------
        # Semantic contribution
        # --------------------------------------------------

        if item["semantic_rank"] is not None:

            score += 1.0 / (
                item["semantic_rank"] + 1
            )

        # --------------------------------------------------
        # Lexical contribution
        # --------------------------------------------------

        if item["lexical_rank"] is not None:

            score += 3.0 / (
                item["lexical_rank"] + 1
            )

        # --------------------------------------------------
        # Actual lexical score contribution
        # --------------------------------------------------

        score += (
            item["lexical_score"] * 0.05
        )

        # --------------------------------------------------
        # Extra calendar ranking
        # --------------------------------------------------

        metadata = item["metadata"]

        document = item["document"].lower()

        source = metadata.get(
            "source",
            ""
        ).lower()

        question_lower = question.lower()

        if "odd semester" in question_lower:

            if "odd" in source:
                score += 3.0

            if "even" in source:
                score -= 3.0

        elif "even semester" in question_lower:

            if "even" in source:
                score += 3.0

            if "odd" in source:
                score -= 3.0

        if (
            "last working day" in question_lower
            or "last day" in question_lower
        ):

            if "closure of class work" in document:
                score += 5.0

        ranked_results.append(
            (
                score,
                item
            )
        )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    ranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # ==================================================
    # FINAL RESULTS
    # ==================================================

    final_results = ranked_results[:top_k]

    return {
        "documents": [
            [
                item["document"]
                for score, item in final_results
            ]
        ],
        "metadatas": [
            [
                item["metadata"]
                for score, item in final_results
            ]
        ]
    }


# ==================================================
# 11. RETRIEVER TEST
# ==================================================

if __name__ == "__main__":

    question = (
        "What is the last working day of the "
        "odd semester for seniors in 2026-27?"
    )

    results = retrieve(
        question,
        top_k=5
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    print(
        "\n================ RETRIEVED CHUNKS ================\n"
    )

    for i, (
        document,
        metadata
    ) in enumerate(
        zip(
            documents,
            metadatas
        ),
        start=1
    ):

        print(
            f"\n--- CHUNK {i} ---"
        )

        print(document)

        print("\nMetadata:")

        print(metadata)

    print(
        "\n===================================================="
    )