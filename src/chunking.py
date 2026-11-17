def chunk_documents(documents, chunk_size=500, overlap=100):
    """
    Split extracted PDF pages or PowerPoint slides into smaller text chunks.

    Each input document should be a dictionary containing:
    - source
    - text
    - page (for PDFs) or slide (for PowerPoints)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        text = document.get("text", "").strip()

        if not text:
            continue

        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunk = {
                "source": document.get("source", "unknown"),
                "text": " ".join(chunk_words),
                "chunk_index": len(chunks),
            }

            # Preserve the original PDF page or PowerPoint slide number.
            if "page" in document:
                chunk["page"] = document["page"]

            if "slide" in document:
                chunk["slide"] = document["slide"]

            chunks.append(chunk)

            # Stop once the final chunk has been created.
            if start + chunk_size >= len(words):
                break

    return chunks