import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import Settings

class RagChain:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings


    def build_qa_chain(self, schema):
   
        vectorstore = Redis.from_existing_index(
            embedding=self.embeddings,
            redis_url=self.settings.REDIS_URL,
            index_name=self.settings.INDEX_NAME,
            schema=schema
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatOpenAI(model=self.settings.OPENAI_MODEL_NAME, api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.OPENAI_BASE_URL)


        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful finance assistant. Use the provided context to answer the question.\n"
                "Cite sources in the format [source - page] where possible. If the answer is not in the context, say 'I am not aware of this.'.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}

        )

        print(f"rag_chain :: created qa_chain : {qa_chain}")
        return qa_chain
