import os
import glob
from app.ingestion.processor import process_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "DATA")

def main():
    print(f"Looking for PDFs in {DATA_DIR}...")
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the DATA directory.")
        return

    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        print(f"Ingesting: {file_name}")
        process_file(
            file_path=file_path,
            file_name=file_name,
            source_type="pdf",
            source_name="Local Data Directory"
        )
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
