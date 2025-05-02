import gradio as gr
from gliner import GLiNER
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
import re
from datasets import load_dataset
from langchain.schema import Document
import os

# Initialize once
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Qdrant setup
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="customer_support_docsv1",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

retriever = doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-scout-17b-16e-instruct")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent assistant. Use context and chat history to answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}")
])

rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["query"]),
    "query": lambda x: x["query"],
    "chat_history": lambda x: x["chat_history"]
}) | chat_prompt | llm | StrOutputParser()

# Shared memory
memory = ChatMessageHistory()

# Gradio handler
def chat_fn(message, history_list):
    # Use LangChain-style history for context
    response = rag_chain.invoke({
        "query": message,
        "chat_history": memory.messages
    })

    # Append new messages to the LangChain memory
    memory.add_user_message(message)
    memory.add_ai_message(response)

    return response

chatbot = gr.ChatInterface(fn=chat_fn, title="🛠️ Customer Support Chatbot")
chatbot.launch()
