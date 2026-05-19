from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()


# Load embedding model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# Load your FAISS index
# (assumes you already created it)

vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)


# Request schema

class Query(BaseModel):
    question: str


# RAG endpoint

@app.post("/chat")
def chat(query: Query):
    docs = vectorstore.similarity_search(query.question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    return {
        "question": query.question,
        "context": context
    }