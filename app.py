import streamlit as st
from src.retrieval.rag import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GITAM College Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 GITAM College Knowledge Assistant")

st.write(
    "Ask questions about GITAM academic regulations, "
    "calendars, syllabi, examinations, and other official documents."
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask your question",
    placeholder="Example: How many credits are required to complete the undergraduate program?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching college documents..."):

            try:

                answer, sources = generate_answer(question)

                # --------------------------------------------
                # ANSWER
                # --------------------------------------------

                st.subheader("Answer")

                st.write(answer)


                # --------------------------------------------
                # SOURCES
                # --------------------------------------------

                if sources:

                    st.subheader("Sources")

                    for i, metadata in enumerate(sources, start=1):

                        source = metadata.get(
                            "source",
                            "Unknown source"
                        )

                        page = metadata.get(
                            "page",
                            "Unknown page"
                        )

                        st.write(
                            f"**{i}.** {source} — Page {page}"
                        )

            except Exception as e:

                st.error(
                    f"Something went wrong while processing your question:\n\n{e}"
                )