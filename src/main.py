from src.config import (
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from src.document_loader import load_pdf
from src.ppt_loader import load_powerpoint
from src.chunking import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import build_vector_store
from src.rag import answer_question


def load_document(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix in {".ppt", ".pptx"}:
        return load_powerpoint(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".pdf", ".ppt", ".pptx"}
    )

    if not files:
        print(f"No PDF or PowerPoint files found in: {DOCUMENTS_DIR}")
        return

    print("\nStudy materials:")
    for number, file_path in enumerate(files, start=1):
        print(f"{number}. {file_path.name}")

    choice = input("\nEnter the number of the file to study: ").strip()

    if not choice.isdigit() or not 1 <= int(choice) <= len(files):
        print("Invalid selection.")
        return

    selected_file = files[int(choice) - 1]

    print(f"\nLoading {selected_file.name}...")
    documents = load_document(selected_file)

    if not documents:
        print("No readable text was found in that file.")
        return

    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        print("No text chunks were created.")
        return

    print(f"Creating embeddings for {len(chunks)} text chunks...")
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    index, chunks = build_vector_store(chunks, embeddings)

    print("\nReady! Ask a question about the material.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = answer_question(
                question=question,
                index=index,
                chunks=chunks,
                top_k=TOP_K,
            )

            print(f"\nAssistant: {answer}\n")

        except Exception as error:
            print(f"\nCould not answer that question: {error}\n")


if __name__ == "__main__":
    main()