import os
import shutil
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
PERSIST_DIRECTORY  = "vector_db"
EMBEDDING_MODEL    = "gemini-embedding-001"

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 100
BATCH_SIZE        = 80
BATCH_WAIT_SECS   = 65


def find_pdfs(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"'{folder}' doesn't exist. Create it and drop your flower PDFs inside, "
            f"then run this script again."
        )

    pdf_paths = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{folder}'. Add at least one PDF "
            f"(e.g. a flowers encyclopedia or a flora-of-India guide) and re-run."
        )

    return pdf_paths


def load_and_split(pdf_paths: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []

    for path in pdf_paths:
        try:
            pages = PyPDFLoader(path).load()
        except Exception as e:
            # A single corrupted or password-protected PDF shouldn't take
            # down the whole ingestion run — skip it and keep going.
            print(f"  Skipping '{os.path.basename(path)}': could not read it ({e})")
            continue

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"  {os.path.basename(path)}: {len(pages)} page(s) -> {len(chunks)} chunk(s)")

    return all_chunks


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to a .env file in the project root "
            "before running ingest.py (this script always reads from .env, never "
            "from Streamlit secrets, since it's run from the terminal)."
        )

    print(f"Scanning '{KNOWLEDGE_BASE_DIR}' for PDFs...")
    pdf_paths = find_pdfs(KNOWLEDGE_BASE_DIR)
    print(f"Found {len(pdf_paths)} PDF(s).\n")

    print("Loading and splitting documents...")
    chunks = load_and_split(pdf_paths)
    if not chunks:
        raise RuntimeError("Every PDF failed to load — nothing to embed. Check the files above.")
    print(f"\nTotal chunks to embed: {len(chunks)}")

    # Rebuild from scratch each time rather than appending. This avoids
    # silently duplicating embeddings for PDFs that haven't changed —
    # simpler to reason about than incremental updates, and this script
    # runs rarely enough that the rebuild cost doesn't matter.
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"\nRemoving existing vector database at '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    print("Embedding chunks with Gemini and writing to Chroma...")
    print(f"  {len(chunks)} chunks  ·  batch size {BATCH_SIZE}  ·  {BATCH_WAIT_SECS}s wait between batches")
    print(f"  Estimated time: ~{((len(chunks) - 1) // BATCH_SIZE) * BATCH_WAIT_SECS} seconds\n")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    # Send first batch — this creates the Chroma collection.
    first_batch = chunks[:BATCH_SIZE]
    print(f"  Batch 1/{(len(chunks) - 1) // BATCH_SIZE + 1}  ({len(first_batch)} chunks)...")
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    # Send remaining batches with a wait between each so the free-tier
    # 100-requests-per-minute quota resets before the next batch hits.
    remaining = chunks[BATCH_SIZE:]
    for batch_num, start in enumerate(range(0, len(remaining), BATCH_SIZE), start=2):
        batch = remaining[start:start + BATCH_SIZE]
        total_batches = (len(chunks) - 1) // BATCH_SIZE + 1

        print(f"  Waiting {BATCH_WAIT_SECS}s for API quota to reset...", end="", flush=True)
        for _ in range(BATCH_WAIT_SECS):
            time.sleep(1)
            print(".", end="", flush=True)
        print()

        print(f"  Batch {batch_num}/{total_batches}  ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

    print(f"\nDone. {len(chunks)} chunks indexed into '{PERSIST_DIRECTORY}'.")
    print("Commit the vector_db/ folder to GitHub so Streamlit Cloud")
    print("has the database ready on startup without re-running this script.")


if __name__ == "__main__":
    main()