"""Configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Default to gpt-4.1 for longer, more stable structured outputs.
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "1234567bpmn")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
    
    # Processing
    CONFIDENCE_THRESHOLD: float = 0.8
    SIMILARITY_MERGE_THRESHOLD: float = 0.90
    SIMILARITY_REVIEW_THRESHOLD: float = 0.80
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Upload / file conversion
    # - If uploaded file is not a PDF, convert to PDF when possible.
    ENABLE_FILE_CONVERSION: bool = os.getenv("ENABLE_FILE_CONVERSION", "true").lower() == "true"
    # Preferred converter: libreoffice/soffice. If empty, we try to find on PATH.
    LIBREOFFICE_PATH: str = os.getenv("LIBREOFFICE_PATH", "")

    # OCR / Vision extraction
    # - If a PDF page contains images, OCR should run even if text exists.
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"
    OCR_ALWAYS_IF_IMAGES: bool = os.getenv("OCR_ALWAYS_IF_IMAGES", "true").lower() == "true"
    # "tesseract" | "openai_vision" (will fallback automatically if deps unavailable)
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract").lower()
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    # Safety limits
    OCR_MAX_PAGES: int = int(os.getenv("OCR_MAX_PAGES", "50"))
    OCR_MAX_IMAGE_PIXELS: int = int(os.getenv("OCR_MAX_IMAGE_PIXELS", str(2000 * 2000)))

    # SOP segmentation (optional but improves multi-process docs)
    # When enabled and OpenAI key is available, detect SOP boundaries and create sections per SOP.
    ENABLE_SOP_SEGMENTATION: bool = os.getenv("ENABLE_SOP_SEGMENTATION", "true").lower() == "true"
    SOP_MAX_PAGES_FOR_BOUNDARY: int = int(os.getenv("SOP_MAX_PAGES_FOR_BOUNDARY", "30"))
    
    # Performance optimization options
    EVIDENCE_MODE: str = os.getenv("EVIDENCE_MODE", "full")  # "full", "reference_only", "off"
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "fixed")  # "fixed", "semantic"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure output directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories
Config.ensure_dirs()




