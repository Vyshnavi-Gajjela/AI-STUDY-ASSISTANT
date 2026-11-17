import streamlit as st

from src.config import TOP_K
from src.quiz import generate_quiz
from src.rag import answer_question, has_document_support
from src.retriever import retrieve_chunks
from src.summarizer import summarize_chunks
from src.upload_processing import process_uploaded_document


st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")


def _clear_document_outputs():
    for key in ("summary", "quiz", "quiz_submitted", "chat_history"):
        st.session_state.pop(key, None)


st.title("📚 Document Study Assistant")
st.write(
    "Upload a PDF or PowerPoint, then ask questions, get a simple summary, "
    "or generate a practice quiz. Answers are grounded in the selected document."
)
st.info(
    "This app is intended to be publicly accessible. Do not upload sensitive or "
    "confidential documents. Document text is sent to the configured AI provider "
    "when generating answers, summaries, or quizzes."
)

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "pptx"],
    help="Supported formats: PDF and PowerPoint (.pptx).",
)

if uploaded_file:
    st.caption(f"Selected document: **{uploaded_file.name}**")
    if st.button("Process document", type="primary"):
        try:
            with st.spinner("Extracting text and creating searchable embeddings..."):
                index, chunks = process_uploaded_document(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not process the document: {error}")
        else:
            st.session_state.document = {
                "name": uploaded_file.name,
                "index": index,
                "chunks": chunks,
            }
            _clear_document_outputs()
            st.success(
                f"Processed {uploaded_file.name}: {len(chunks)} searchable chunks."
            )

document = st.session_state.get("document")
if not document:
    st.info("Upload and process a PDF or PPTX document to get started.")
    st.stop()

st.subheader(f"Studying: {document['name']}")
st.caption(
    f"{len(document['chunks'])} chunks indexed with exact-keyword and semantic search."
)

chat_tab, summary_tab, quiz_tab = st.tabs(["Ask a question", "Document summary", "Quiz"])

with chat_tab:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Ask something about this document")
        submitted = st.form_submit_button("Get answer")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching the document and preparing your answer..."):
                    results = retrieve_chunks(
                        question=question,
                        index=document["index"],
                        chunks=document["chunks"],
                        top_k=TOP_K,
                    )
                    answer = answer_question(
                        question=question,
                        index=document["index"],
                        chunks=document["chunks"],
                        top_k=TOP_K,
                        retrieved_results=results,
                    )
            except (RuntimeError, ValueError) as error:
                st.error(f"Could not answer the question: {error}")
            else:
                sources = results if has_document_support(results) else []
                history = st.session_state.setdefault("chat_history", [])
                history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                })

    for item in reversed(st.session_state.get("chat_history", [])):
        st.markdown(f"**You asked:** {item['question']}")
        st.markdown(item["answer"])
        with st.expander("Document passages used"):
            for result in item["sources"]:
                chunk = result["chunk"]
                location = chunk.get("page", chunk.get("slide"))
                where = f", page/slide {location}" if location is not None else ""
                st.caption(f"{chunk.get('source', document['name'])}{where}")
                st.write(chunk["text"])

with summary_tab:
    st.write("Create a plain-language summary covering the selected document.")
    if st.button("Generate full summary"):
        try:
            with st.spinner("Summarizing the document..."):
                st.session_state.summary = summarize_chunks(document["chunks"])
        except (RuntimeError, ValueError) as error:
            st.error(f"Could not summarize the document: {error}")
    if st.session_state.get("summary"):
        st.markdown(st.session_state.summary)

with quiz_tab:
    question_count = st.slider("Number of questions", min_value=5, max_value=15, value=5)
    if st.button("Generate quiz"):
        try:
            with st.spinner(f"Creating {question_count} questions from the document..."):
                st.session_state.quiz = generate_quiz(document["chunks"], question_count)
                st.session_state.quiz_submitted = False
        except (RuntimeError, ValueError) as error:
            st.error(f"Could not generate the quiz: {error}")

    quiz = st.session_state.get("quiz")
    if quiz:
        with st.form("quiz_form"):
            responses = {}
            for number, item in enumerate(quiz, start=1):
                st.markdown(f"**{number}. {item['question']}**")
                responses[number - 1] = st.radio(
                    "Choose one answer",
                    options=list(item["options"]),
                    format_func=lambda key, options=item["options"]: (
                        f"{key}. {options[key]}"
                    ),
                    index=None,
                    key=f"quiz_answer_{number}",
                    label_visibility="collapsed",
                )
            quiz_submitted = st.form_submit_button("Submit answers")

        if quiz_submitted:
            st.session_state.quiz_submitted = True
            st.session_state.quiz_responses = responses

        if st.session_state.get("quiz_submitted"):
            responses = st.session_state.quiz_responses
            score = 0
            for number, item in enumerate(quiz, start=1):
                chosen = responses.get(number - 1)
                correct = chosen == item["correct_answer"]
                score += int(correct)
                if chosen is None:
                    answer_key = item["correct_answer"]
                    st.error(
                        f"{number}. Not answered. Correct answer: {answer_key}. "
                        f"{item['options'][answer_key]}\n\n{item['explanation']}"
                    )
                elif correct:
                    st.success(f"{number}. Correct — {item['explanation']}")
                else:
                    answer_key = item["correct_answer"]
                    st.error(
                        f"{number}. Your answer: {chosen}. {item['options'][chosen]}\n\n"
                        f"Correct answer: {answer_key}. {item['options'][answer_key]}\n\n"
                        f"{item['explanation']}"
                    )
            st.info(f"Your score: {score} / {len(quiz)}")
