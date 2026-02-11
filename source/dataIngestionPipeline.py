import re
from pathlib import Path
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


# ================= CONFIG =================

LOG_DIR = "../assets/BCC_CORP_FCC_CreacionCuentaAPP_log_files"   # folder shown in your screenshot
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "weblogic_logs"

KEYWORDS = [
    "STUCK",
    "HOGGED",
    "ThreadPool",
    "HEALTH_CRITICAL",
    "HEALTH_WARN",
    "Service Bus",
    "ExecuteThread",
    "Exception",
    "ERROR",
    "WARN"
]

# =========================================


def is_relevant(line: str) -> bool:
    return any(k.lower() in line.lower() for k in KEYWORDS)


def extract_thread(line: str) -> str:
    """
    Extract ExecuteThread if present.
    Example: [ExecuteThread: '12']
    """
    match = re.search(r"ExecuteThread[^'\"]*['\"]?(\d+)['\"]?", line)
    return f"ExecuteThread-{match.group(1)}" if match else "unknown"


def chunk_file(file_path: Path) -> List[Document]:
    documents = []
    buffer = []
    current_thread = None

    with file_path.open(errors="ignore") as f:
        for line in f:
            if not is_relevant(line):
                continue

            thread = extract_thread(line)

            # New execution context → flush previous chunk
            if current_thread and thread != current_thread and buffer:
                documents.append(
                    Document(
                        page_content="".join(buffer),
                        metadata={
                            "file": file_path.name,
                            "thread": current_thread,
                            "type": file_path.suffix
                        }
                    )
                )
                buffer = []

            current_thread = thread
            buffer.append(line)

        # flush remaining
        if buffer:
            documents.append(
                Document(
                    page_content="".join(buffer),
                    metadata={
                        "file": file_path.name,
                        "thread": current_thread,
                        "type": file_path.suffix
                    }
                )
            )

    return documents


def ingest_all_logs(log_dir: Path):
    embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},      # or "cuda" if available
                    encode_kwargs={"normalize_embeddings": True}
                )
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    all_docs = []

    for file_path in log_dir.iterdir():
        if file_path.suffix not in [".log", ".out"]:
            continue

        print(f"Processing {file_path.name}")
        docs = chunk_file(file_path)
        all_docs.extend(docs)

    if all_docs:
        vectordb.add_documents(all_docs)

    print(f"Inserted {len(all_docs)} chunks into Chroma DB")



# if __name__ == "__main__":
#     ingest_all_logs(Path("./uploaded_logs/BCC_CORP_FCC_CreacionCuentaAPP_log_files/BCC_CORP_FCC_CreacionCuentaAPP_log_files"))
