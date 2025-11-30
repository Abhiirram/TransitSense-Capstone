import json
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ✅ NEW CORRECT IMPORT
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

# --------------------- TRANSIT RETRIEVER ---------------------

class TransitRetriever:
    def __init__(self, json_path="train_data.json", policy_path="policies.txt"):
        self.json_path = json_path
        self.policy_path = policy_path
        self.vectorstore = None
        self.retriever = None

        # Build RAG pipeline now
        self._build_pipeline()

    # -------- Load Train JSON Documents --------
    def _load_json_docs(self):
        """Loads structured train data"""
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)

            docs = []
            for t in data:
                text = (
                    f"Train {t['name']} ({t['train_no']}) travels from {t['source']} "
                    f"to {t['destination']}. Price: {t['price']}. Class: {t['class']}. "
                    f"Days: {t['days']}."
                )
                docs.append(
                    Document(page_content=text, metadata={"source": "schedule"})
                )
            return docs

        except FileNotFoundError:
            print(f"⚠️ Warning: {self.json_path} not found.")
            return []

    # -------- Load Policies Text Documents --------
    def _load_text_docs(self):
        try:
            with open(self.policy_path, "r") as f:
                text = f.read()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=80
            )
            docs = splitter.create_documents([text])

            for d in docs:
                d.metadata["source"] = "policy"

            return docs

        except FileNotFoundError:
            print(f"⚠️ Warning: {self.policy_path} not found.")
            return []

    # -------- Build Vectorstore Pipeline --------
    def _build_pipeline(self):
        print("⚙️ Building RAG Pipeline...")

        train_docs = self._load_json_docs()
        policy_docs = self._load_text_docs()

        all_docs = train_docs + policy_docs

        if not all_docs:
            print("❌ No documents found to index.")
            return

        embeddings = OpenAIEmbeddings()

        self.vectorstore = FAISS.from_documents(all_docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        print(f"✅ RAG Pipeline Ready. Loaded {len(all_docs)} chunks.")

    # -------- Search Function (Used by Tools) --------
    def search(self, query: str):
        if not self.retriever:
            return "Pipeline not initialized."

        print(f"🔎 Searching: {query}")
        docs = self.retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
