"""Configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv


def _load_env_chain() -> None:
    project_root = Path(__file__).resolve().parents[2]
    project_env = project_root / ".env"
    if project_env.exists():
        load_dotenv(dotenv_path=project_env, override=False)


_load_env_chain()


class Config:
    """Application configuration."""
    
    # OpenAI-compatible API settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").strip()
    OCR_BASE_URL: str = (os.getenv("OCR_BASE_URL") or LLM_BASE_URL).strip()
    EMBEDDING_BASE_URL: str = (os.getenv("EMBEDDING_BASE_URL") or LLM_BASE_URL).strip()

    # Models
    # Default to gpt-4.1 for longer, more stable structured outputs.
    LLM_MODEL: str = (os.getenv("LLM_MODEL") or "gpt-4.1").strip()
    OCR_MODEL: str = (os.getenv("OCR_MODEL") or LLM_MODEL).strip()
    EMBEDDING_MODEL: str = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    EMBEDDING_DIMENSIONS: int = int(
        os.getenv(
            "EMBEDDING_DIMENSIONS",
            "2560" if "qwen3-embedding-4b" in EMBEDDING_MODEL.lower() else "1536",
        )
    )
    EMBEDDING_TIMEOUT_SEC: float = float(os.getenv("EMBEDDING_TIMEOUT_SEC", "60"))
    
    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "").strip()
    NEO4J_USER: str = os.getenv("NEO4J_USER", "").strip()
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "").strip()
    
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
    # "tesseract" | "openai_vision" | "synap"
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract").lower()
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    # Safety limits
    OCR_MAX_PAGES: int = int(os.getenv("OCR_MAX_PAGES", "50"))
    OCR_MAX_IMAGE_PIXELS: int = int(os.getenv("OCR_MAX_IMAGE_PIXELS", str(2000 * 2000)))
    SYNAP_OCR_BASE_URL: str = os.getenv("SYNAP_OCR_BASE_URL", "").strip()
    SYNAP_OCR_API_KEY: str = os.getenv("SYNAP_OCR_API_KEY", "").strip()
    SYNAP_OCR_POLL_INTERVAL_SEC: float = float(os.getenv("SYNAP_OCR_POLL_INTERVAL_SEC", "1"))
    SYNAP_OCR_TIMEOUT_SEC: float = float(os.getenv("SYNAP_OCR_TIMEOUT_SEC", "120"))

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

    @classmethod
    def assert_neo4j_config(
        cls,
        uri: str = "",
        user: str = "",
        password: str = "",
    ) -> tuple[str, str, str]:
        """Validate required Neo4j settings and return normalized values."""
        effective_uri = (uri or cls.NEO4J_URI or "").strip()
        effective_user = (user or cls.NEO4J_USER or "").strip()
        effective_password = (password or cls.NEO4J_PASSWORD or "").strip()

        missing = []
        if not effective_uri:
            missing.append("NEO4J_URI")
        if not effective_user:
            missing.append("NEO4J_USER")
        if not effective_password:
            missing.append("NEO4J_PASSWORD")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Missing required Neo4j configuration: {joined}. "
                "Set all required environment variables before starting pdf2bpmn."
            )

        # In container/Kubernetes environments, localhost almost always means wrong target.
        if os.getenv("KUBERNETES_SERVICE_HOST") and "localhost" in effective_uri:
            raise RuntimeError(
                f"Invalid NEO4J_URI for Kubernetes: {effective_uri}. "
                "Use a reachable service endpoint (e.g. neo4j+s://... or service DNS), not localhost."
            )

        return effective_uri, effective_user, effective_password


# Initialize directories
Config.ensure_dirs()




