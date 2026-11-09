import sys
import uvicorn

# Configure UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("=" * 60)
    print("⚡ Starting Agentic RAG Server...")
    print("🌐 Dashboard UI: http://localhost:8000")
    print("📚 API Docs:     http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
