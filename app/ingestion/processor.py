import os
import sys
import uuid
import logfire
from app.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.services.retrival.embedding import embed_batch, get_embedding_dim
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.pdf import parse_pdf 
from app.ingestion.loaders.office import parse_docx

logfire.configure(service_name="Ingestion Processor", level=settings.LOG_LEVEL)
PROCESSOR_DIR="processed_data"
client=QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=True
)
def save_data_locally(data:dict,source_type:str,source_name:str)->str:
    """
    Saves the chunk data in locally.
    Returns the path to the saved file.
    """
    
    if not os.path.exists(PROCESSOR_DIR):
        os.makedirs(PROCESSOR_DIR)
    file_name=f"{source_type}_{source_name}_{uuid.uuid4().hex}.txt"
    file_path=os.path.join(PROCESSOR_DIR,file_name)
    with open(file_path,'w',encoding='utf-8') as f:
        f.write(data['text'])
    return file_path
def process_file(file_path:str,file_name:str,source_type:str,source_name:str)->None:
    """
    Processes a file based on its source type and saves the processed data to Qdrant.
    """
    logfire.info(f"Processing file: {file_name} of type: {source_type} from source: {source_name}")
    try:
        if source_type=="pdf":
            text=parse_pdf(file_path)
        elif source_type=="docx":
            text=parse_docx(file_path)
        elif source_type=="html":
            text=parse_html(file_path)
        elif source_type=="text":
            text=parse_text(file_path)
        else:
            logfire.error(f"Unsupported source type: {source_type}")
            return
        
        if not text.strip():
            logfire.warning(f"No text extracted from {file_name}. Skipping.")
            return
        
        data={"text":text}
        saved_file_path=save_data_locally(data,source_type,source_name)
        
        # Embedding the text
        embeddings=embed_batch([text])
        
        # Saving to Qdrant
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[0],
                    payload={
                        "file_name":file_name,
                        "source_type":source_type,
                        "source_name":source_name,
                        "local_path":saved_file_path
                    }
                )
            ]
        )
        
        logfire.info(f"Successfully processed and saved {file_name} to Qdrant.")
    except Exception as e:
        logfire.error(f"Failed to process {file_name}: {e}")