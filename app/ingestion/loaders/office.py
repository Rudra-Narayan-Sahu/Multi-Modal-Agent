import os
import logfire

def parse_office(file_path: str) -> str:
    """
    Parses Office documents (.docx, .pptx) using docx/pptx libraries or unstructured.
    """
    ext = os.path.splitext(file_path)[1].lower()
    with logfire.span("Office Document Parsing", filename=file_path):
        # 1. Try python-docx for .docx
        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if full_text.strip():
                    return full_text
            except Exception as e:
                logfire.warning(f"docx library parsing failed: {e}")

        # 2. Try python-pptx for .pptx
        if ext == ".pptx":
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                text_runs = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            text_runs.append(shape.text)
                full_text = "\n".join(text_runs)
                if full_text.strip():
                    return full_text
            except Exception as e:
                logfire.warning(f"pptx library parsing failed: {e}")

        # 3. Fallback to unstructured
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements if str(el).strip()])
            return full_text
        except Exception as e:
            logfire.error(f"Office parse failed: {e}")
            raise e

