import os
import sys

# Configure UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

from app.ingestion.processor import process_all_files

def main():
    print("=" * 60)
    print(">>> STARTING AGENTIC RAG DOCUMENT INGESTION")
    print("=" * 60)
    
    result = process_all_files("DATA")
    
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY:")
    print(f"Total Files Found:   {result.get('total_files', 0)}")
    print(f"Total Chunks Stored: {result.get('total_chunks', 0)}")
    
    if result.get("processed_files"):
        print("\nProcessed Files:")
        for pf in result["processed_files"]:
            print(f"  [+] {pf['file_name']}: {pf['chunks']} chunks")
            
    if result.get("errors"):
        print("\nErrors:")
        for err in result["errors"]:
            print(f"  [-] {err['file_name']}: {err['error']}")
            
    print("=" * 60)

if __name__ == "__main__":
    main()


