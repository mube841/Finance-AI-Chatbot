from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def load_and_chunk_pdf(path: str, orig_file_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    
    docs: List[Document] = []
    try:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

        for d in pages:
            meta = dict(d.metadata or {})

            page_num = meta.get("page") or meta.get("page_number") or 0
            meta["page"] = int(page_num)
            meta["source"] = orig_file_name

            chunks = text_splitter.split_documents([d])
            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk.page_content.strip(),
                        metadata=meta.copy()
                    )
                )

        print(f"Generated {len(docs)} chunks")
        return docs

    except Exception as e:
        print(f"load_and_chunk_pdf failed : {e}")
        return []
        
    
