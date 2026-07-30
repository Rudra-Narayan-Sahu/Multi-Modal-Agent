from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip(): 
            return []
            
        # Use LangChain's recursive splitter which correctly falls back 
        # from double newlines, to single newlines, to spaces, to characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks