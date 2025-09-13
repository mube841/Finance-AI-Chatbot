FinanceDoc Q&A Assistant – AI Chatbot for Financial Reports (LangChain + Redis + Streamlit)

This project is a Retrieval-Augmented Generation (RAG) chatbot for financial documents.
Upload PDFs like annual reports, earnings call transcripts, or financial statements.
The system chunks & embeds text using HuggingFace embeddings and stores them in Redis VectorStore.
Ask natural language questions in a Streamlit web app, and get accurate answers with citations (source + page).

This is fully packaged with Docker Compose, so one command is needed to run demo locally.

This repo showcases how businesses can convert unstructured finance documents into an interactive, AI-powered Q&A assistant — applicable in finance, legal, HR, and enterprise knowledge management.
