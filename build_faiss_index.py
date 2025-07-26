# build_faiss_index.py

import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ——— Load .env ———
load_dotenv(find_dotenv())
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDFS_PATH            = "data/"            # your PDF folder
OUTPUT_PATH          = "vectorstore/db_faiss"

def load_all_pdfs(folder: str):
    docs = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, fn))
            for doc in loader.load_and_split():
                doc.metadata["source"] = fn
                docs.append(doc)
    if not docs:
        raise RuntimeError(f"No PDFs found in {folder}")
    return docs

def main():
    # 1) load docs
    docs = load_all_pdfs(PDFS_PATH)

    # 2) embed & build FAISS
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db  = FAISS.from_documents(docs, emb)

    # 3) save
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    db.save_local(OUTPUT_PATH)
    print(f"✅ FAISS index built (d={db.index.d}) at {OUTPUT_PATH}")

if __name__=="__main__":
    main()
