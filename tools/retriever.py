from __future__ import annotations 

import os
from typing import List, Tuple
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)

try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True  # PDF loading is available if dependencies are installed
except Exception:
    PDF_AVAILABLE = False


def _load_documents(sample_docs_dir: str) -> List[Document]:
    """
    Loads documents from data/sample_docs/.
    Supports .txt by default; supports PDF if pypdf and PyPDFLoader are installed.
    """
    docs: List[Document] = []

    # Load text files recursively from the directory
    txt_loader = DirectoryLoader(
        sample_docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=True,
    )
    docs.extend(txt_loader.load())

    # Load PDFs if the optional PDF loader is available
    if PDF_AVAILABLE:
        pdf_paths: List[str] = []
        for root, _, files in os.walk(sample_docs_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, f))
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    return docs


def _split_documents(docs: List[Document]) -> List[Document]:
    # Split documents into overlapping chunks for retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = splitter.split_documents(docs)

    # Attach stable citation metadata for traceability in downstream agents
    per_doc_counter = {}
    for global_i, d in enumerate(split_docs):
        src = d.metadata.get("source", "unknown_source")
        doc_id = os.path.basename(src)

        per_doc_counter.setdefault(doc_id, 0)
        local_i = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1

        page = d.metadata.get("page", None)
        d.metadata["doc_id"] = doc_id
        d.metadata["chunk_id"] = local_i
        d.metadata["global_chunk_id"] = global_i

        # Human-readable location string used by citations (page/chunk if available)
        if page is not None:
            d.metadata["location"] = f"page {page}, chunk {local_i}"
        else:
            d.metadata["location"] = f"chunk {local_i}"

    return split_docs


def get_vectorstore(
    persist_dir: str,
    collection_name: str = "supplychain_copilot",
) -> Chroma:
    # Create/load a persistent Chroma collection with OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def build_or_update_index(
    sample_docs_dir: str,
    persist_dir: str,
    collection_name: str = "supplychain_copilot",
) -> Tuple[Chroma, int]:
    """
    Rebuilds the Chroma index from scratch every time (prevents duplicates).
    Returns (vectorstore, num_chunks_indexed).
    """
    os.makedirs(sample_docs_dir, exist_ok=True)

    raw_docs = _load_documents(sample_docs_dir)
    if not raw_docs:
        os.makedirs(persist_dir, exist_ok=True)
        vs = get_vectorstore(persist_dir, collection_name)
        return vs, 0

    # Remove existing persistent index to avoid duplicate chunks across rebuilds
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    split_docs = _split_documents(raw_docs)

    # Create a fresh vectorstore and add chunked documents
    vs = get_vectorstore(persist_dir, collection_name)
    vs.add_documents(split_docs)

    return vs, len(split_docs)

def retrieve(
    query: str,
    persist_dir: str,
    k: int = 6,
    collection_name: str = "supplychain_copilot",
) -> List[Document]:
    """
    Retrieve top-k docs from persistent Chroma store.
    """
    vs = get_vectorstore(persist_dir, collection_name)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    # Newer LangChain retrievers are Runnables
    docs = retriever.invoke(query)

    # Deduplicate results by (doc_id, location) to avoid repeated chunks
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("doc_id"), d.metadata.get("location"))
        if key not in seen:
            seen.add(key)
            unique.append(d)

    # Back-compat: ensure list[Document]
    return unique
