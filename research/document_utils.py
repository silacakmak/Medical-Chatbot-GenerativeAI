# document_utils.py

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_pdf_loader():
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader

def load_pdf_files(data_path):
    PDFLoader = get_pdf_loader()
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PDFLoader)
    return loader.load()

def text_split(documents, chunk_size=500, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
