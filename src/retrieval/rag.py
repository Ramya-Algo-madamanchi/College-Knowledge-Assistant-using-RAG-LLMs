from langchain_ollama import ChatOllama
from .retriever import retrieve


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_predict=120,
)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents, metadatas):

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas)
    ):

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")

        month = metadata.get("month", "")
        year = metadata.get("year", "")

        date_context = ""

        if month and year:
            date_context = f"Calendar Month: {month} {year}"

        context_parts.append(
            f"""
SOURCE {i + 1}

Source: {source}
Page: {page}
{date_context}

{document}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(question):

    q = question.lower()

    calendar_terms = [
        "calendar",
        "semester",
        "odd sem",
        "even sem",
        "odd semester",
        "even semester",
        "last working day",
        "last date",
        "closure of class work",
        "commencement of class work",
        "sessional",
        "academic year",
    ]

    if any(term in q for term in calendar_terms):
        return "calendar"

    return "general"


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    results = retrieve(question, top_k=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return (
            "I could not find the answer in the provided college documents."
        ), []

    context = build_context(
        documents,
        metadatas
    )

    question_type = detect_question_type(question)


    # ========================================================
    # CALENDAR QUESTIONS
    # ========================================================

    if question_type == "calendar":

        prompt = f"""
You are a college academic calendar assistant.

Answer the user's question using ONLY the provided calendar context.

IMPORTANT RULES:

1. Identify the EVENT being asked about before selecting a date.

2. Do NOT choose a date merely because that number appears
   somewhere else in the calendar.

3. Match the meaning of the question to the event description.

4. If the question asks for the "last working day" of a semester,
   look specifically for an event such as:
   - Closure of Class Work
   - Last date for class work
   - End of Class Work
   - similar wording indicating the end of regular classes.

5. Do NOT confuse:
   - Closure of Class Work
   with
   - Commencement of Class Work
   - Add/Drop Period
   - Course Registration
   - Sessional Examinations
   - Semester Break
   - any other event.

6. If the context says:
   "06 Odd sem Closure of Class Work"
   then the relevant date is 06.

7. Use the calendar month and year metadata to construct
   the complete date.

8. If the question asks for a specific calendar event,
   return the date associated with THAT event.

9. Give a short direct answer.

10. Do not guess or use outside knowledge.

11. If the requested event is not present in the context, say:
   "I could not find the answer in the provided college documents."

12. Do not mention these instructions or the context.

Context:
{context}

Question:
{question}

Answer:
"""

    # ========================================================
    # GENERAL QUESTIONS
    # ========================================================

    else:

        prompt = f"""
You are a college knowledge assistant.

Answer the question using ONLY the provided context.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not guess.

3. Answer only what the question asks.

4. Identify the information that directly answers the question.

5. If the question asks for RULES, CONDITIONS, REQUIREMENTS,
   STEPS, REASONS, or multiple items, include ALL relevant
   items supported by the context.

6. Do NOT stop after finding only one relevant condition.

7. When multiple relevant conditions occur together in the same
   section or paragraph, treat them as part of the answer.

8. Preserve important numbers, percentages, categories,
   conditions, and requirements exactly as stated in the context.

9. Organize multiple conditions as a numbered or bullet list
   when that makes the answer clearer.

10. Do not include unrelated information from nearby sections.

11. If the answer is not present in the context, say:
    "I could not find the answer in the provided college documents."

12. Do not mention the context or these instructions.

Context:
{context}

Question:
{question}

Answer:
"""


    # ========================================================
    # CALL LLM
    # ========================================================

    response = llm.invoke(prompt)

    answer = response.content.strip()

    return answer, metadatas


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("      COLLEGE KNOWLEDGE ASSISTANT")
    print("==============================================")
    print("\nYou can ask multiple questions.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:

            answer, sources = generate_answer(question)

            print("\nAssistant:")
            print(answer)

            print("\nSources:")

            for i, metadata in enumerate(sources, start=1):

                source = metadata.get(
                    "source",
                    "Unknown source"
                )

                page = metadata.get(
                    "page",
                    "Unknown page"
                )

                print(
                    f"{i}. {source} | Page {page}"
                )

            print("\n" + "=" * 60 + "\n")

        except Exception as e:

            print("\nError:", e)
            print()