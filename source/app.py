"""
app.py
------
Orchestration layer for:
1. Log ingestion
2. Incident analysis (RAG)

Designed to be reused by Streamlit or API layer.
"""

from pathlib import Path

# Import your existing modules
from dataIngestionPipeline import ingest_all_logs
from RAG import generate_incident_report
import shutil
from pathlib import Path



# =========================================================
# INGESTION ENTRY POINT
# =========================================================

def run_ingestion(logs_dir: str):
    """
    Ingest logs from a given directory into Chroma DB
    """
    logs_path = Path(logs_dir)
    print(f"logs path::{logs_path}")

    if not logs_path.exists() or not logs_path.is_dir():
        raise ValueError(f"Invalid logs directory: {logs_dir}")

    ingest_all_logs(logs_path)

    return {
        "status": "success",
        "message": f"Logs ingested successfully from {logs_dir}"
    }


# =========================================================
# RAG ENTRY POINT
# =========================================================

def run_incident_analysis(ticket_text: str):
    """
    Run RAG pipeline on ticket text
    """
    if not ticket_text or not ticket_text.strip():
        raise ValueError("Ticket text cannot be empty")

    result = generate_incident_report(ticket_text)
    return result


# =========================================================
# CLI TEST (OPTIONAL)
# =========================================================

# if __name__ == "__main__":

#     # ---- Example usage ----

#     LOGS_DIR = "../assets/BCC_CORP_FCC_CreacionCuentaAPP_log_files"

#     TICKET_TEXT = """
#     Weblogic Server reports hogged threads and stuck threads.
#     Service Bus Kernel entered HEALTH_CRITICAL state.
#     """

#     print("Running ingestion...")
#     ingestion_result = run_ingestion(LOGS_DIR)
#     print(ingestion_result)

#     print("\nRunning incident analysis...")
#     analysis_result = run_incident_analysis(TICKET_TEXT)
#     print(analysis_result)

def clear_chromadb():
    """
    Completely clears the ChromaDB persistent storage.
    USE WITH CAUTION.
    """
    chroma_path = Path("./chroma_db")

    if chroma_path.exists() and chroma_path.is_dir():
        shutil.rmtree(chroma_path)
        return {
            "status": "success",
            "message": "ChromaDB cleared successfully"
        }
    else:
        return {
            "status": "noop",
            "message": "ChromaDB directory does not exist"
        }

