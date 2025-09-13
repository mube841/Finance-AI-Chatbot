import os

from langchain.vectorstores.redis import Redis
from utils.pdf_loader import load_and_chunk_pdf
from config import Settings
import redis


class DataIngestor:

    def __init__(self, embeddings=None):
        self.embeddings = embeddings
    
    def ingest_files(self, file_paths, orig_file_name, schema: dict, settings: Settings):
       
        redis_url = settings.REDIS_URL

        all_docs = []
        for p in file_paths:
            print(f"Ingesting {p} ...")
            docs = load_and_chunk_pdf(p, orig_file_name)
            for d in docs:
                
                if "source" not in d.metadata:
                    d.metadata["source"] = os.path.basename(p)
                    
            all_docs.extend(docs)

        if not all_docs:
            print("No documents to ingest.")
            return

        print(f"Ingesting {len(all_docs)} chunks into Redis...")
        print(f"all_docs : {all_docs}")
        
        r = redis.from_url(redis_url, decode_responses=True)
        index_name = settings.INDEX_NAME
        
        Redis.drop_index(index_name=index_name, delete_documents=True, redis_url=settings.REDIS_URL)

        existing_indexes = r.execute_command("FT._LIST")
        print(f"existing_indexes : {existing_indexes}")
        
       
        if index_name in existing_indexes:
            print(f"Index '{index_name}' exists. Loading from existing index...")
            vectorstore = Redis.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=redis_url,
                schema=schema
            )
            print(f"loaded vectorstore : {vectorstore}")
        else:
            print(f"Index '{index_name}' does not exist. Creating new index...")

            vectorstore = Redis.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                redis_url=redis_url,
                index_name=index_name
            )



        print(f"Ingested {len(all_docs)} chunks into Redis index '{index_name}'.")
        return vectorstore
