from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List


file_path = "C:\\Users\\ZUBAIR\\Downloads\\Doc Retriever\\EAP.pdf"

ball =[]
if len(ball)==0: 
    loader =PyPDFLoader(file_path)

    docs=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits= text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)

ball.append(1)

def retriever(query: str) -> List[Document]:
    answer = vector_store.similarity_search(query,k=3)
    output =[answer[0].page_content,answer[1].page_content,answer[2].page_content]
    return output


