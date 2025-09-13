import streamlit as st
import tempfile
from ingestion.data_ingestor import DataIngestor
from config import Settings
from app.rag_chain import RagChain



class AppUI:
    def __init__(self, data_ingestor: DataIngestor, settings: Settings, embeddings=None, index_schema=dict):
        self.data_ingestor = data_ingestor
        self.settings = settings
        self.index_schema = index_schema
        self.embeddings = embeddings
        self.rag_chain = RagChain(settings, embeddings)

    def create_ui(self ):
        st.set_page_config(page_title="Finance RAG Assistant", layout="wide")
        st.title("Finance Docs Assistant")
        st.header("Ask a question")

        self.populate_sidebar()
        self.process_query()

        st.markdown("---")
        st.markdown("**Notes**: This demo uses OpenAI embeddings & chat model by default. You can swap providers in the code.")

    def populate_sidebar(self):
        with st.spinner("Uploading PDFs..."):
            try:
                uploaded_files = st.sidebar.file_uploader("Upload PDF(s) to index", type=["pdf"], accept_multiple_files=True)
                if uploaded_files:
                    self.process_upload(uploaded_files=uploaded_files, index_schema=self.index_schema)
            except Exception as e:
                st.sidebar.error(f"Upload failed: {e}")
            

        st.sidebar.markdown("---")
        st.sidebar.header(f"Index: {self.settings.INDEX_NAME}")
        


    def process_upload(self, uploaded_files: list, index_schema: dict):

        if st.sidebar.button("Ingest uploaded PDFs"):
            saved_paths = []
            for up in uploaded_files:

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(up.getbuffer())
                tmp.flush()
                saved_paths.append(tmp.name)
            with st.spinner("Ingesting PDFs into Redis..."):
                try:
                    self.data_ingestor.ingest_files(saved_paths, up.name, index_schema, self.settings)
                    st.sidebar.success("Ingestion complete.")
                except Exception as e:
                    print(f"ingestion failed, error : {e}")
                    st.sidebar.error(f"Ingestion failed: {e}")

    def process_query(self):
        query = st.text_input("Enter your question about the uploaded finance docs")

        if st.button("Ask") and query:
            with st.spinner("Retrieving answer..."):
                try:
                    qa = self.rag_chain.build_qa_chain(self.index_schema)
                    result = qa({"query": query})
                    answer = result.get("result") or result.get("answer") or ""
                    st.subheader("Answer")
                    st.write(answer)

                    src_docs = result.get("source_documents") or result.get("source_documents", [])
                    if src_docs:
                        st.subheader("Sources / Retrieved chunks")
                        for i, d in enumerate(src_docs):
                            meta = d.metadata if hasattr(d, "metadata") else {}
                            source = meta.get("source", "unknown")
                            page = meta.get("page") or meta.get("page_number") or meta.get("page_index") or meta.get("page_label", "na")
                            st.markdown(f"**{i + 1}. Source:** `{source}` — page: `{page}`")
                            with st.expander("Show chunk text"):
                                st.write(d.page_content)
                except Exception as e:
                    st.error(f"Query failed: {e}")