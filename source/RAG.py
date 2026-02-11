import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
import logging
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import timezone
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


# =========================================================
# ENV + GROQ SETUP
# =========================================================

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=api_key)


def send_to_groq(
    prompt,
    model_name="llama-3.3-70b-versatile",
    max_tokens=6000,
    temperature=0.4,
):
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# =========================================================
# CHROMA CONFIG
# =========================================================

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "weblogic_logs"


# =========================================================
# STEP 1: TICKET → MULTIPLE SEARCH QUERIES
# =========================================================

def decompose_ticket_to_queries(ticket_text: str):
    """
    Convert ticket into multiple focused log-retrieval queries
    """

    base_queries = [
        "stuck threads ExecuteThread",
        "hogged threads ThreadPoolRuntime",
        "ThreadPoolRuntime stuck threads",
        "Service Bus Kernel HEALTH_CRITICAL",
        "Service Bus stuck threads",
    ]

    llm_prompt = f"""
Extract 3 to 5 concise technical log search queries from the incident below.
Return ONLY a JSON array of strings.

Incident:
{ticket_text}
"""

    try:
        response = send_to_groq(llm_prompt, temperature=0.2)
        llm_queries = json.loads(response)
        logger.info("--------------------------------------------------")
        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] PREPARED QUERIES")
        for q in list(set(base_queries + llm_queries)):
            logger.info(f"- {q}")
        logger.info("--------------------------------------------------")

        return list(set(base_queries + llm_queries))
    except Exception:
        logger.info("--------------------------------------------------")
        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] PREPARED QUERIES (FALLBACK)")
        for q in base_queries:
            logger.info(f"- {q}")
        logger.info("--------------------------------------------------")

        return base_queries


# =========================================================
# STEP 2: MULTI-QUERY VECTOR RETRIEVAL
# =========================================================

def retrieve_relevant_logs(ticket_text: str, k_per_query: int = 4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    queries = decompose_ticket_to_queries(ticket_text)

    all_docs = []
    for q in queries:
        docs = vectordb.similarity_search(q, k=k_per_query)
        all_docs.extend(docs)

    return deduplicate_docs(all_docs)


# =========================================================
# STEP 3: DEDUPLICATE BY EXECUTION CONTEXT
# =========================================================

def deduplicate_docs(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        key = (
            doc.metadata.get("file"),
            doc.metadata.get("thread"),
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    logger.info("==================================================")
    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] FINAL RETRIEVED LOG CHUNKS")
    logger.info(f"Total chunks: {len(unique_docs)}")

    for idx, doc in enumerate(unique_docs, start=1):
        logger.info("--------------------------------------------------")
        logger.info(f"Chunk #{idx}")
        logger.info(f"File   : {doc.metadata.get('file')}")
        logger.info(f"Thread : {doc.metadata.get('thread')}")
        logger.info(f"Type   : {doc.metadata.get('type')}")
        logger.info("Content:")
        logger.info(doc.page_content[:1000])  # limit to avoid huge logs

    logger.info("==================================================")
    return unique_docs


# =========================================================
# STEP 4: EVIDENCE-FIRST PROMPT
# =========================================================

def build_prompt(ticket_text: str, retrieved_docs):
    log_blocks = []

    for doc in retrieved_docs:
        log_blocks.append(
            f"[File: {doc.metadata.get('file')} | Thread: {doc.metadata.get('thread')}]\n"
            f"{doc.page_content}"
        )

    logs_combined = "\n\n".join(log_blocks)

    prompt = f"""
You are a senior WebLogic / Oracle Service Bus production engineer.

Incident Description:
{ticket_text}

Below are log excerpts grouped by execution thread.
Each group represents a single execution path.

Logs:
{logs_combined}

TASK:
1. Identify the specific technical issue observed in the logs.
2. Write an incident summary strictly based on log evidence.
3. Produce action points that DIRECTLY address the identified issue.
   - Each action must map to observed evidence.
   - NO generic best practices.
   - If evidence is insufficient, explicitly say so.

Output JSON ONLY:
{{
  "identified_issue": "",
  "incident_summary": [],
  "action_points": []
}}
"""
    return prompt.strip()


# =========================================================
# STEP 5: END-TO-END RAG EXECUTION
# =========================================================

def extract_json_from_response(text: str):
    """
    Extract JSON object from LLM response that may contain markdown fences.
    """
    try:
        # Case 1: raw JSON
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Case 2: JSON inside ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Case 3: JSON inside ``` ... ```
    match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError("No valid JSON found in LLM response")


def generate_incident_report(ticket_text: str):
    retrieved_docs = retrieve_relevant_logs(ticket_text)

    if not retrieved_docs:
        return {
            "identified_issue": "Insufficient log evidence",
            "incident_summary": ["No relevant logs found"],
            "action_points": ["Verify log ingestion and keyword filters"],
        }

    prompt = build_prompt(ticket_text, retrieved_docs)
    response = send_to_groq(prompt)

    try:
        return extract_json_from_response(response)
    except Exception:
        return {"raw_response": response}



# =========================================================
# MAIN
# =========================================================

# if __name__ == "__main__":

#     ticket_summary = """
# summary 1: Weblogic Server: thread_pool:hoggedThreads.active The number of hogged threads is 17.

# summary 2: FAULT: ME$WLS_HealthStatus_OverallHealthStatus
# Component:threadpool, State:HEALTH_WARN, ReasonCode:[ThreadPool has stuck threads]

# summary 3: Weblogic Server: ME$WLS_HealthStatus_OverallHealthStatus
# Component:Service Bus Kernel(Application), State:HEALTH_CRITICAL,
# ReasonCode:[STUCK_THREADS]
# """

#     result = generate_incident_report(ticket_summary)
#     print(json.dumps(result, indent=2))
