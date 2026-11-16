
import streamlit as st

from src.rag import answer_question


st.set_page_config(
    page_title="Enterprise HR Assistant",
    page_icon="🏢",
    layout="centered",
)


st.title("🏢 Enterprise HR Assistant")

st.markdown(
    """
    Ask questions about the sample HR policies.

    The assistant retrieves relevant policy sections and
    generates answers based on the available documents.
    """
)

st.info(
    "This is an experimental demo using sample HR policies. "
    "It is not a real company's HR system."
)


SAMPLE_QUESTIONS = [
    "How many annual leave days do employees receive?",
    "How many days in advance should I request annual leave?",
    "Can unused annual leave be carried over?",
    "How many days per week can employees work remotely?",
    "When is a medical certificate required for sick leave?",
    "What is the deadline for submitting expense claims?",
    "Does the company reimburse personal expenses?",
    "Does the policy explain paid parental leave?",
]


st.subheader("Try a sample question")

for question in SAMPLE_QUESTIONS:
    if st.button(question, use_container_width=True):
        st.session_state.question = question


st.divider()

question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="For example: How many days can I work remotely?",
    height=100,
    max_chars=500,
)

if st.button(
    "Ask HR Assistant",
    type="primary",
    use_container_width=True,
):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        # Basic per-session usage limit.
        request_count = st.session_state.get("request_count", 0)

        if request_count >= 5:
            st.warning(
                "You have reached the demo limit of 5 questions "
                "in this session."
            )

        else:
            try:
                with st.spinner("Searching HR documents..."):
                    result = answer_question(question.strip())

                st.session_state.request_count = request_count + 1

                st.subheader("Answer")
                st.markdown(result["answer"])

                st.caption(
                    f"Questions used in this session: "
                    f"{st.session_state.request_count}/5"
                )

                st.subheader("Retrieved sources")

                if result["sources"]:
                    for index, source in enumerate(
                        result["sources"],
                        start=1,
                    ):
                        with st.expander(
                            f"{index}. {source['source']} — "
                            f"{source['section_title']}"
                        ):
                            st.caption(
                                f"Retrieval score: "
                                f"{source['score']:.4f}"
                            )
                            st.write(source["text"])

                else:
                    st.info("No relevant source was retrieved.")

            except Exception as error:
                st.error(
                    "The assistant could not process your question. "
                    "Please try again later."
                )
                st.caption(f"Technical detail: {type(error).__name__}")

