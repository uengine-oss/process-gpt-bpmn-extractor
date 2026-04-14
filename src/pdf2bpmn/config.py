"""Configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv


def _load_env_chain() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    root_env = repo_root / ".env"
    project_env = project_root / ".env"

    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)
    if project_env.exists():
        load_dotenv(dotenv_path=project_env, override=True)


_load_env_chain()


class Config:
    """Application configuration."""
    
    # Unified LLM proxy settings (OpenAI-compatible API)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip()
    LLM_PROXY_URL: str = os.getenv("LLM_PROXY_URL", "").strip()
    LLM_PROXY_API_KEY: str = os.getenv("LLM_PROXY_API_KEY", "").strip()

    # OpenAI
    OPENAI_API_KEY: str = LLM_PROXY_API_KEY or os.getenv("OPENAI_API_KEY", "")
    # Default to gpt-4.1 for longer, more stable structured outputs.
    OPENAI_MODEL: str = LLM_MODEL or os.getenv("OPENAI_MODEL", "gpt-4.1")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # OpenAI-compatible endpoint (e.g., OpenRouter: https://openrouter.ai/api/v1)
    OPENAI_BASE_URL: str = LLM_PROXY_URL or os.getenv("OPENAI_BASE_URL", "")
    
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
    # Temporary bypass switch for local tests:
    # when true, force EXTRACT LLM input sections to exactly one section.
    FORCE_SINGLE_SECTION: bool = os.getenv("FORCE_SINGLE_SECTION", "false").lower() == "true"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure output directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories
Config.ensure_dirs()




