# # connect_memory_with_llm.py

# import os
# from dotenv import load_dotenv, find_dotenv

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # ——— Load .env ———
# load_dotenv(find_dotenv())
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("HF_TOKEN not set")

# # ——— Config ———
# HUGGINGFACE_REPO_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# DB_FAISS_PATH        = "vectorstore/db_faiss"
# TOP_K                = 3

# # ——— Prepare LLM, Embeddings & Prompt ———
# def load_llm():
#     ep = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         huggingfacehub_api_token=HF_TOKEN,
#         temperature=0.5,
#         max_new_tokens=512,
#         task="conversational",
#     )
#     return ChatHuggingFace(llm=ep)

# PROMPT = PromptTemplate(
#     template="""\
# Use only the provided context to answer the question. If you don’t know, say you don’t know.

# Context: {context}
# # Question: {question}
# # """,
# #     input_variables=["context", "question"],
# # )

# # # ——— Main ———
# # if __name__ == "__main__":
# #     # load embedding & FAISS
# #     emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# #     db  = FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)

# #     # build QA chain
# #     qa = RetrievalQA.from_chain_type(
# #         llm=load_llm(),
# #         chain_type="stuff",
# #         retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
# #         return_source_documents=True,
# #         chain_type_kwargs={"prompt": PROMPT},
# #     )

# #     query = input("Enter query: ").strip()
# #     result = qa.invoke({"query": query})

# #     print("\n=== RESULT ===")
# #     print(result["result"])

# #     print("\n=== SOURCES ===")
# #     for d in result["source_documents"]:
# #         src = d.metadata.get("source", "<no source>")
# #         txt = d.page_content.replace("\n", " ").strip()
# #         print(f"- {src}: {txt[:200]}…")
# # # …all your existing imports and setup…

# # # ——— Exposed Function ———
# # def qa(query: str) -> dict:
# #     """
# #     Perform a QA query over the FAISS-backed vector store.
# #     Returns a dict with keys 'result' and 'source_documents'.
# #     """
# #     return qa_chain.invoke({"query": query})

# # # If you ever run this file directly, make sure you still load the index:
# # if __name__ == "__main__":
# #     q = input("Enter query: ")
# #     resp = qa(q)
# #     print("Answer:", resp["result"])
# #     for d in resp["source_documents"]:
# #         print("-", d.metadata.get("source", "no source"), d.page_content[:200])

# # connect_memory_with_llm.py

# import os
# from dotenv import load_dotenv, find_dotenv

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # ——— Load .env ———
# load_dotenv(find_dotenv())
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("HF_TOKEN not set in .env or system environment.")

# # ——— Config ———
# HUGGINGFACE_REPO_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# DB_FAISS_PATH        = "vectorstore/db_faiss"
# TOP_K                = 3

# # ——— Prepare LLM ———
# def load_llm():
#     endpoint = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         huggingfacehub_api_token=HF_TOKEN,
#         temperature=0.5,
#         max_new_tokens=512,
#         task="conversational",
#     )
#     return ChatHuggingFace(llm=endpoint)

# # ——— Load Embeddings & FAISS Index ———
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# # ——— Prepare Prompt ———
# PROMPT = PromptTemplate(
#     template="""
# Use only the provided context to answer the question. If you don’t know, say you don’t know.

# Context: {context}
# Question: {question}
# """,
#     input_variables=["context", "question"],
# )

# # ——— Build RetrievalQA Chain ———
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT},
# )

# # ——— Exposed Function ———
# def qa(query: str) -> dict:
#     """
#     Perform a QA query over the FAISS-backed vector store.
#     Returns a dict with keys 'result' and 'source_documents'.
#     """
#     return qa_chain.invoke({"query": query})

# # ——— CLI Mode ———
# if __name__ == "__main__":
#     q = input("Enter query: ")
#     resp = qa(q)
#     print("\n=== RESULT ===")
#     print(resp.get("result", ""))
#     print("\n=== SOURCES ===")
#     for d in resp.get("source_documents", []):
#         src = d.metadata.get("source", "no source")
#         txt = d.page_content.replace("\n", " ")[:200]
#         print(f"- {src}: {txt}…")

# import os
# from dotenv import load_dotenv, find_dotenv

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # ——— Load .env ———
# load_dotenv(find_dotenv())
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("HF_TOKEN not set in .env or system environment.")

# # ——— Config ———
# HUGGINGFACE_REPO_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# DB_FAISS_PATH        = "vectorstore/db_faiss"
# TOP_K                = 3

# # ——— Prepare LLM ———
# def load_llm():
#     endpoint = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         huggingfacehub_api_token=HF_TOKEN,
#         temperature=0.5,
#         max_new_tokens=512,
#         task="conversational",
#     )
#     return ChatHuggingFace(llm=endpoint)

# # ——— Load Embeddings & FAISS Index ———
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# # ——— Prepare Prompt ———
# PROMPT = PromptTemplate(
#     template="""
# Use only the provided context to answer the question. If you don’t know, say you don’t know.

# Context: {context}
# Question: {question}
# """,
#     input_variables=["context", "question"],
# )

# # ——— Build RetrievalQA Chain ———
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT},
# )

# # ——— Exposed Function ———
# def qa(query: str) -> dict:
#     """
#     Perform a QA query over the FAISS-backed vector store.
#     Returns a dict with keys 'result' and 'source_documents'.
#     """
#     return qa_chain.invoke({"query": query})

# # ——— CLI Mode ———
# if __name__ == "__main__":
#     q = input("Enter query: ")
#     resp = qa(q)
#     print("\n=== RESULT ===")
#     print(resp.get("result", ""))
#     print("\n=== SOURCES ===")
#     for d in resp.get("source_documents", []):
#         src = d.metadata.get("source", "no source")
#         txt = d.page_content.replace("\n", " ")[:200]
#         print(f"- {src}: {txt}…")







# # File: connect_memory_with_llm.py
# import os
# import re
# from dotenv import load_dotenv, find_dotenv



# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # ——— Load .env ———
# load_dotenv(find_dotenv())
# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("⚠️ HF_TOKEN not set in .env or system environment.")

# # ——— Config ———
# HUGGINGFACE_REPO_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# DB_FAISS_PATH        = "vectorstore/db_faiss"
# TOP_K                = 3

# # ——— Prepare LLM ———
# def load_llm():
#     endpoint = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         huggingfacehub_api_token=HF_TOKEN,
#         temperature=0.0,        # deterministic
#         max_new_tokens=512,
#         task="text-generation",
#     )
#     return ChatHuggingFace(llm=endpoint)

# # ——— Load Embeddings & FAISS Index ———
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# # ——— Prepare Prompt ———
# PROMPT = PromptTemplate(
#     template="""
# Use only the provided context to answer the question. If you don’t know, say you don’t know.

# Context:
# {context}

# Question:
# {question}
# """,
#     input_variables=["context", "question"],
# )

# # ——— Build RetrievalQA Chain ———
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT},
# )

# # ——— Exposed Function ———
# def qa(query: str) -> dict:
#     """
#     Perform a QA query over the FAISS-backed vector store.
#     Returns a dict with keys 'result' and 'source_documents'.
#     """
#     # Retrieve top-K docs
#     docs = db.as_retriever(search_kwargs={"k": TOP_K}).get_relevant_documents(query)
#     # Determine if any doc mentions the query terms
#     terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
#     found = any(
#         term in doc.page_content.lower()
#         for term in terms
#         for doc in docs
#     )
#     if not found:
#         return {"result": "I don't know.", "source_documents": []}
#     # Otherwise, use the QA chain
#     return qa_chain.invoke({"query": query})

# # ——— CLI Mode ———
# if __name__ == "__main__":
#     q = input("Enter query: ")
#     resp = qa(q)
#     print("\n=== RESULT ===")
#     print(resp.get("result", ""))
#     print("\n=== SOURCES ===")
#     for d in resp.get("source_documents", []):
#         src = d.metadata.get("source", "no source")
#         txt = d.page_content.replace("\n", " ")[:200]
#         print(f"- {src}: {txt}…")






# File: connect_memory_with_llm.py
import os
import re
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# ——— Load environment ———
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in .env or system environment.")

# ——— Configuration ———
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"
TOP_K = 3

# ——— Initialize LLM ———
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.0,
        max_new_tokens=512,
        task="text-generation",
    )
    return ChatHuggingFace(llm=endpoint)

# ——— Load Embeddings & FAISS Index ———
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# ——— Prepare Prompt & RetrievalQA Chain ———
prompt = PromptTemplate(
    template="""
Use only the provided context to answer the question. If you don’t know, say you don’t know.

Context: {context}
Question: {question}
""",
    input_variables=["context", "question"],
)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# ——— Exposed QA Function ———
def qa(query: str) -> dict:
    """
    If relevant context is found in the FAISS index, return the RetrievalQA result (answer + sources).
    Otherwise, return 'I don't know.' with no sources.
    """
    # Retrieve top-K relevant docs
    docs = db.as_retriever(search_kwargs={"k": TOP_K}).get_relevant_documents(query)
    # Simple term matching to check relevance
    terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
    relevant = any(
        term in doc.page_content.lower()
        for term in terms
        for doc in docs
    )
    if not relevant:
        return {"result": "I don't know.", "source_documents": []}

    # If relevant, use RetrievalQA to get answer and sources
    res = qa_chain.invoke({"query": query})
    return {
        "result": res.get("result", ""),
        "source_documents": res.get("source_documents", [])
    }

# ——— CLI Mode ———
if __name__ == "__main__":
    question = input("Enter query: ")
    response = qa(question)
    print("Answer:\n", response["result"])
    if response["source_documents"]:
        print("\nSources:")
        for doc in response["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"- {src}: {snippet}…")
