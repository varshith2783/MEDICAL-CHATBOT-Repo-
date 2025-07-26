# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# import os
# from dotenv import load_dotenv, find_dotenv

# # Load environment variables from .env
# load_dotenv(find_dotenv())

# # Now HF_TOKEN should be available in the environment
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN not found in environment. Please set it in your .env file or system environment.")


# print("Hugging Face Token: ", HF_TOKEN)



# # Step 1: Load raw PDF(s)
# DATA_PATH="data/"
# def load_pdf_files(data):
#     loader = DirectoryLoader(data,
#                              glob='*.pdf',
#                              loader_cls=PyPDFLoader)
    
#     documents=loader.load()
#     return documents

# documents=load_pdf_files(data=DATA_PATH)
# #print("Length of PDF pages: ", len(documents))



# # Step 1: Load raw PDF(s)
# # Step 2: Create Chunks
# def create_chunks(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
#                                                  chunk_overlap=50)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks

# text_chunks=create_chunks(extracted_data=documents)
# #print("Length of Text Chunks: ", len(text_chunks))
# # Step 3: Create Vector Embeddings 

# def get_embedding_model():
#     embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return embedding_model
# #all-MiniLM-L6-v2 SEMANTIC SEARCH MAPS SENTENCES AND PARAGRAPHS TO A 384 DIMENSIONAL DENSE CVECTOR SPACE AND CAN BE USED FOR TASKS LIKE CLUSTERING OR SEMANTIC SEARCH
# embedding_model=get_embedding_model()

# # Step 4: Store embeddings in FAISS
# DB_FAISS_PATH="vectorstore/db_faiss"
# #IN THE ABOVE PATH THE DATABASE WILL BE STORED I.E EMBEDDINGS OF CHUNKS WILL BE STORED
# db=FAISS.from_documents(text_chunks, embedding_model)
# db.save_local(DB_FAISS_PATH)
# Aggregated LLM Chatbot Code Snippets
# This file collects all code segments you provided until you wrote 'THE END'.

# --- File: connect_memory_with_llm.py ---


# import os
# from dotenv import load_dotenv, find_dotenv

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # ——— Load .env ———
# load_dotenv(find_dotenv())
# HF_TOKEN = os.environ.get("HF_TOKEN")
# print(f"Hugging Face Token: {HF_TOKEN}")  # debug
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
# Question: {question}
# """,
#     input_variables=["context", "question"],
# )

# # ——— Main ———
# if __name__ == "__main__":
#     print("Loading embeddings and FAISS index...")
#     emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     db  = FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)
#     print(f"FAISS index loaded from '{DB_FAISS_PATH}'.")

#     qa = RetrievalQA.from_chain_type(
#         llm=load_llm(),
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT},
#     )

#     query = input("Enter query: ").strip()
#     print(f"Query received: {query}")
#     result = qa.invoke({"query": query})

#     print("\n=== RESULT ===")
#     print(result["result"])

#     print("\n=== SOURCES ===")
#     for d in result["source_documents"]:
#         src = d.metadata.get("source", "<no source>")
#         txt = d.page_content.replace("\n", " ").strip()
#         print(f"- {src}: {txt[:200]}…")

# # --- File: create_memory_for_llm.py ---

# import os
# from dotenv import load_dotenv, find_dotenv

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # ——— Main Script with Logging ———
# def main():
#     # Load environment variables from .env
#     load_dotenv(find_dotenv())
#     HF_TOKEN = os.environ.get("HF_TOKEN")
#     print(f"Hugging Face Token: {HF_TOKEN}")
#     if not HF_TOKEN:
#         raise ValueError("HF_TOKEN not found in environment. Please set it in your .env or system environment.")

#     # Step 1: Load raw PDF(s)
#     DATA_PATH = "data/"
#     loader = DirectoryLoader(
#         DATA_PATH,
#         glob='*.pdf',
#         loader_cls=PyPDFLoader
#     )
#     documents = loader.load()
#     print(f"Loaded {len(documents)} document(s) from '{DATA_PATH}'.")

#     # Step 2: Create Chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     text_chunks = splitter.split_documents(documents)
#     print(f"Created {len(text_chunks)} text chunk(s).")

#     # Step 3: Create Vector Embeddings
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Step 4: Store embeddings in FAISS
#     DB_FAISS_PATH = "vectorstore/db_faiss"
#     db = FAISS.from_documents(text_chunks, embedding_model)
#     db.save_local(DB_FAISS_PATH)
#     print(f"Saved FAISS index to '{DB_FAISS_PATH}'. Congratulations!")

# if __name__ == "__main__":
#     main()


# # --- File: streamlit_app.py ---
# import os
# from dotenv import load_dotenv, find_dotenv
# import streamlit as st

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # ——— Load environment ———
# load_dotenv(find_dotenv())
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     st.error("HF_TOKEN not set. Please add it to your .env file.")
#     st.stop()

# # ——— Config ———
# HUGGINGFACE_REPO_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# DB_FAISS_PATH        = "vectorstore/db_faiss"
# TOP_K                = 3

# # ——— Load components ———
# @st.cache_resource
# def get_llm():
#     ep = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         huggingfacehub_api_token=HF_TOKEN,
#         temperature=0.5,
#         max_new_tokens=512,
#         task="conversational",
#     )
#     return ChatHuggingFace(llm=ep)

# @st.cache_resource
# def get_db():
#     emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     return FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)

# # Prepare prompt
# PROMPT = PromptTemplate(
#     template="""\
# Use only the provided context to answer the question. If you don’t know, say you don’t know.

# Context: {context}
# Question: {question}
# """,
#     input_variables=["context", "question"],
# )

# # Build chain
# llm = get_llm()
# db  = get_db()
# qa  = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT},
# )

# # ——— Streamlit UI ———
# st.title("Medical Chatbot")
# user_query = st.text_input("Ask me about medical topics:")

# if st.button("Submit") and user_query:
#     with st.spinner("Thinking..."):
#         res = qa.invoke({"query": user_query})
#     st.subheader("Answer")
#     st.write(res["result"])
#     st.subheader("Sources")
#     for doc in res["source_documents"]:
#         src = doc.metadata.get("source", "<no source>")
#         st.markdown(f"**{src}**: {doc.page_content[:200]}…")

# # --- Notes ---
# # - To build embeddings: `python create_memory_for_llm.py`
# # - To run the web UI: `streamlit run streamlit_app.py`
# # - Ensure `.env` contains `HF_TOKEN` for Hugging Face API access.

import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # Load environment
    load_dotenv(find_dotenv())
    HF_TOKEN = os.environ.get("HF_TOKEN")
    print(f"Hugging Face Token: {HF_TOKEN}")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment. Please set it in .env or system environment.")

    # Step 1: Load raw PDFs
    DATA_PATH = "data/"
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from '{DATA_PATH}'.")

    # Step 2: Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunk(s).")

    # Step 3: Compute embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Store in FAISS
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Saved FAISS index to '{DB_FAISS_PATH}'. Congratulations!")

if __name__ == "__main__":
    main()
