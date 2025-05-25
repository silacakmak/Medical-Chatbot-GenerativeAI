# prepare_data.py

from weaviate_config import connect_to_weaviate, initialize_embeddings
from document_utils import load_pdf_files, text_split
import os
import sys

def prepare_and_upload_data(data_path):
    # Load documents
    documents = load_pdf_files(data_path)
    if not documents:
        print("No documents loaded.")
        sys.exit(1)

    chunks = text_split(documents)
    embeddings = initialize_embeddings()
    client = connect_to_weaviate()

    class_name = "medibot"

    try:
        collection = client.collections.get(class_name)
        print(f"Collection '{class_name}' already exists.")
    except:
        # Schema creation
        collection = client.collections.create(
            name=class_name,
            description="Medical document chunks for retrieval",
            vectorizer_config=None,
            properties=[
                {"name": "content", "data_type": "text"},
                {"name": "source", "data_type": "text"},
                {"name": "page", "data_type": "number"},
            ],
        )
        print("Schema created.")

    # Add documents
    with collection.batch.fixed_size(batch_size=50) as batch:
        for i, chunk in enumerate(chunks, 1):
            content = chunk.page_content
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            vector = embeddings.embed_query(content)

            batch.add_object(
                properties={"content": content, "source": source, "page": page},
                vector=vector
            )

    client.close()
    print("✓ Data preparation and upload completed.")

if __name__ == "__main__":
    data_path = "Data"  # veya input ile al
    prepare_and_upload_data(data_path)
