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

# # ——— Prepare prompt ———
# PROMPT = PromptTemplate(
#     template="""\
# Use only the provided context to answer the question. If you don’t know, say you don’t know.

# Context: {context}
# Question: {question}
# """,
#     input_variables=["context", "question"],
# )

# # ——— Build chain ———
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

# File: connect_memory_with_llm.py

# # import os
# # from dotenv import load_dotenv, find_dotenv

# # from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
# # from langchain_core.prompts import PromptTemplate
# # from langchain.chains import RetrievalQA
# # from langchain_community.vectorstores import FAISS

# # # ——— Load environment ———
# # load_dotenv(find_dotenv())
# # HF_TOKEN = os.environ.get("HF_TOKEN")
# # if not HF_TOKEN:
# #     raise RuntimeError("HF_TOKEN not set in .env or system environment.")

# # # ——— Configuration ———
# # HUGGINGFACE_REPO_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
# # EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# # DB_FAISS_PATH        = "vectorstore/db_faiss"
# # TOP_K                = 3

# # # ——— Initialize LLM ———
# # def load_llm():
# #     endpoint = HuggingFaceEndpoint(
# #         repo_id=HUGGINGFACE_REPO_ID,
# #         huggingfacehub_api_token=HF_TOKEN,
# #         temperature=0.5,
# #         max_new_tokens=512,
# #         task="conversational",
# #     )
# #     return ChatHuggingFace(llm=endpoint)

# # # ——— Prepare Prompt ———
# # PROMPT = PromptTemplate(
# #     template="""
# # Use only the provided context to answer the question. If you don’t know, say you don’t know.

# # Context: {context}
# # Question: {question}
# # """,
# #     input_variables=["context", "question"],
# # )

# # # ——— Load Embeddings & FAISS ———
# # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# # db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# # # ——— Build RetrievalQA Chain ———
# # qa_chain = RetrievalQA.from_chain_type(
# #     llm=load_llm(),
# #     chain_type="stuff",
# #     retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
# #     return_source_documents=True,
# #     chain_type_kwargs={"prompt": PROMPT},
# # )

# # # ——— Exposed Function ———
# # def qa(query: str) -> dict:
# #     """
# #     Perform a QA query over the FAISS-backed vector store.
# #     Returns a dict with keys 'result' and 'source_documents'.
# #     """
# #     return qa_chain.invoke({"query": query})


# # # File: streamlit_app.py
# # import os
# # from dotenv import load_dotenv, find_dotenv
# # import streamlit as st

# # from connect_memory_with_llm import qa

# # # ——— Streamlit Page Config ———
# # st.set_page_config(page_title="Medical Chatbot", layout="wide")
# # st.title("🩺 Medical Chatbot")
# # st.markdown("Ask me anything about **medical topics** based on your uploaded knowledge base.")

# # # ——— Initialize Chat History ———
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # # ——— User Input ———
# # user_input = st.text_input("You:", key="input")
# # if st.button("Send") and user_input:
# #     with st.spinner("Thinking..."):
# #         resp = qa(user_input)
# #     # Append to history
# #     st.session_state.chat_history.append({
# #         "question": user_input,
# #         "answer": resp.get("result", ""),
# #         "sources": resp.get("source_documents", [])
# #     })

# # # ——— Chat Display Styles ———
# # st.markdown("""
# # <style>
# # .chat-bubble {background-color: #f1f0f0; padding: 1rem; border-radius: 1rem; margin: 1rem 0;}
# # .chat-user {font-weight: bold; margin-bottom: .2rem;}
# # .chat-assistant {margin-top: .5rem;}
# # </style>
# # """, unsafe_allow_html=True)

# # # ——— Render History ———
# # for turn in st.session_state.chat_history:
# #     st.markdown(f"<div class='chat-bubble'><div class='chat-user'>You:</div> {turn['question']}<div class='chat-assistant'><div class='chat-user'>Bot:</div> {turn['answer']}</div></div>", unsafe_allow_html=True)
# #     with st.expander("Show Sources"):
# #         for idx, doc in enumerate(turn["sources"], 1):
# #             st.markdown(f"**Source {idx}:** `{doc.metadata.get('source','Unknown')}`")
# #             st.write(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

# import os
# from dotenv import load_dotenv, find_dotenv
# import streamlit as st

# from connect_memory_with_llm import qa

# st.set_page_config(page_title="Medical Chatbot", layout="wide")

# load_dotenv(find_dotenv())
# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     st.error("⚠️ HF_TOKEN missing. Add it to .env and restart.")
#     st.stop()

# st.title("🩺 Medical Chatbot")
# st.markdown("Ask me anything about **medical topics** based on your uploaded knowledge base.")

# if "history" not in st.session_state:
#     st.session_state.history = []

# with st.form(key="question_form", clear_on_submit=True):
#     user_input = st.text_input("You:", key="input_text")
#     submit = st.form_submit_button("Send")

# if submit and user_input:
#     with st.spinner("Thinking..."):
#         try:
#             response = qa(user_input)
#             answer = response.get("result", "")
#             sources = response.get("source_documents", [])
#         except Exception as e:
#             st.error(f"Error: {e}")
#         else:
#             st.session_state.history.append({"q": user_input, "a": answer, "s": sources})

# for turn in st.session_state.history:
#     st.markdown(f"**You:** {turn['q']}")
#     st.markdown(f"**Bot:** {turn['a']}")
#     with st.expander("Show sources", expanded=False):
#         for idx, doc in enumerate(turn['s'], 1):
#             src = doc.metadata.get("source", "Unknown")
#             content = doc.page_content.strip().replace("\n", " ")
#             st.markdown(f"{idx}. `{src}`: {content[:300]}{'...' if len(content) > 300 else ''}")

import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from connect_memory_with_llm import qa

# ——— Page Config ———
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# ——— Load & validate token ———
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ HF_TOKEN missing. Add it to .env and restart.")
    st.stop()

# ——— UI Header ———
st.title("🩺 Medical Chatbot")
st.markdown("Ask me anything about **medical topics** based on your uploaded knowledge base.")

# ——— Initialize chat history ———
if "history" not in st.session_state:
    st.session_state.history = []

# ——— Input form ———
with st.form(key="question_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input_text")
    submit = st.form_submit_button("Send")

# ——— Handle submission ———
if submit and user_input:
    with st.spinner("Thinking..."):
        try:
            response = qa(user_input)
            answer = response.get("result", "")
            sources = response.get("source_documents", [])
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            st.session_state.history.append({"q": user_input, "a": answer, "s": sources})

# ——— Render chat ———
for turn in st.session_state.history:
    st.markdown(f"**You:** {turn['q']}")
    st.markdown(f"**Bot:** {turn['a']}")
    with st.expander("Show sources", expanded=False):
        for idx, doc in enumerate(turn['s'], 1):
            src = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"{idx}. `{src}`: {content[:300]}{'...' if len(content) > 300 else ''}")
