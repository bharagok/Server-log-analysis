import re
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime

# ================= CONFIG =================
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "weblogic_logs"
OUTPUT_FILE = f"chroma_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def export_all_documents():
    """Export all documents from Chroma DB to structured text file"""
    
    # Initialize with same embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Load Chroma DB
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    print("Fetching all documents from Chroma DB...")
    all_data = vectordb.get()
    
    total_docs = len(all_data['ids'])
    print(f"Found {total_docs} documents")
    
    with Path(OUTPUT_FILE).open('w', encoding='utf-8') as f:
        f.write(f"CHROMA DB EXPORT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
        f.write(f"Collection: {COLLECTION_NAME}\n")
        f.write(f"Total Documents: {total_docs}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        if all_data['metadatas']:
            files = set(meta.get('file', 'unknown') for meta in all_data['metadatas'])
            threads = set(meta.get('thread', 'unknown') for meta in all_data['metadatas'])
            
            f.write("SUMMARY\n")
            f.write(f"Unique Files: {len(files)}\n")
            f.write(f"Unique Threads: {len(threads)}\n")
            f.write(f"Files: {', '.join(sorted(list(files))[:10])}{'...' if len(files) > 10 else ''}\n")
            f.write(f"Threads: {len(threads)} total\n\n")
        
        # Export all documents
        f.write("DOCUMENTS\n")
        f.write("-" * 40 + "\n")
        
        for i, (doc_id, content, metadata) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'] or [{}] * total_docs)):
            f.write(f"\n[{i+1}/{total_docs}] DOCUMENT ID: {doc_id[:12]}...\n")
            
            # Metadata section
            f.write("METADATA:\n")
            if metadata:
                for key, value in metadata.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write("  No metadata\n")
            
            f.write("\nCONTENT:\n")
            f.write("-" * 30 + "\n")
            f.write(content)
            f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ Exported {total_docs} documents to {OUTPUT_FILE}")
    print(f"File size: {Path(OUTPUT_FILE).stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    export_all_documents()
