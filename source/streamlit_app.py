import streamlit as st
import zipfile
import tempfile
import os
import json
from pathlib import Path
import shutil


# Import your backend APIs
from app import run_ingestion, run_incident_analysis
UPLOAD_BASE_DIR = "uploaded_logs"
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)


st.set_page_config(
    page_title="WebLogic Log Analyzer",
    layout="wide"
)

st.title("🔍 WebLogic / OSB Incident Analyzer")

# =========================================================
# SECTION 1: LOG INGESTION
# =========================================================

st.header("1️⃣ Upload Log Files (ZIP)")

uploaded_zip = st.file_uploader(
    "Upload a ZIP file containing WebLogic / OSB logs",
    type=["zip"]
)

ingestion_status = None
logs_dir_path = None

if uploaded_zip:
    with st.spinner("Extracting logs..."):
        upload_dir = os.path.join(UPLOAD_BASE_DIR, uploaded_zip.name.replace(".zip", ""))
        # Clean existing directory if re-uploading same zip
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)

        os.makedirs(upload_dir, exist_ok=True)
        zip_path = os.path.join(upload_dir, uploaded_zip.name)

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(upload_dir)

        logs_dir_path = upload_dir

        st.success(f"Logs extracted to: {logs_dir_path}")

    if st.button("🚀 Run Log Ingestion"):
        with st.spinner("Ingesting logs into vector database..."):
            try:
                ingestion_status = run_ingestion(logs_dir_path)
                st.success("Log ingestion completed successfully")
                # st.json(ingestion_status)
            except Exception as e:
                st.error("Log ingestion failed")
                st.exception(e)

st.divider()

# =========================================================
# SECTION 2: INCIDENT ANALYSIS
# =========================================================

st.header("2️⃣ Incident Analysis")

ticket_text = st.text_area(
    "Paste ticket / alert details here",
    height=200,
    placeholder="Paste ticket summaries, alerts, or incident description..."
)

if st.button("🔎 Analyze Incident"):
    if not ticket_text.strip():
        st.warning("Please enter ticket text before submitting")
    else:
        with st.spinner("Analyzing logs and generating incident report..."):
            try:
                result = run_incident_analysis(ticket_text)

                st.success("Incident analysis completed")

                # -------------------------
                # Display Results
                # -------------------------
                st.subheader("🧠 Identified Issue")
                st.write(result.get("identified_issue", "N/A"))

                st.subheader("📌 Incident Summary")
                for point in result.get("incident_summary", []):
                    st.markdown(f"- {point}")

                st.subheader("🛠️ Action Points")
                for idx, action in enumerate(result.get("action_points", []), start=1):
                    st.markdown(f"**{idx}.** {action}")

                # Raw fallback
                if "raw_response" in result:
                    st.subheader("⚠️ Raw Model Output")
                    st.code(result["raw_response"])

            except Exception as e:
                st.error("Incident analysis failed")
                st.exception(e)

st.divider()

# =========================================================
# FOOTER
# =========================================================

from app import clear_chromadb

if st.button("🗑️ Clear Vector Database"):
    result = clear_chromadb()
    st.success(result["message"])

st.caption("Powered by MiniLM + Chroma + Groq | Evidence-driven RAG")
