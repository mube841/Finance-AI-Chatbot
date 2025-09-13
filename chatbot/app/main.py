from sentence_transformers import SentenceTransformer
from ingestion.data_ingestor import DataIngestor
from langchain_huggingface import HuggingFaceEmbeddings

from config import Settings
from app.ui.appui import AppUI


settings = Settings()
embeddings = HuggingFaceEmbeddings(model_name=settings.HUGGINGFACE_MODEL_NAME, model_kwargs={"device": "cpu"}  )

data_ingestor = DataIngestor(embeddings)

index_schema = {
        "tag": [{"name": "source"}],
        "text": [{"name": "page_label"}],
    }

appui = AppUI(data_ingestor, settings, embeddings, index_schema)
appui.create_ui()
