# weaviate_config.py

import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

def connect_to_weaviate():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
        additional_config=AdditionalConfig(timeout=Timeout(insert=300))
    )
    if not client.is_ready():
        raise ConnectionError("Weaviate is not ready.")
    return client

def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
