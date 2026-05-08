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

    # Memento (process-gpt-memento) - 단일 source-of-truth로 사용한다.
    # - 프론트(메인 채팅)는 항상 메멘토 경유로 파일 업로드/임베딩을 수행하며,
    #   pdf2bpmn은 다운로드/임베딩을 별도로 하지 않고 이 URL을 통해 청크/임베딩을 가져온다.
    MEMENTO_BASE_URL: str = (os.getenv("MEMENTO_BASE_URL") or "http://localhost:8005").rstrip("/")
    MEMENTO_TIMEOUT_SEC: float = float(os.getenv("MEMENTO_TIMEOUT_SEC", "60"))

    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "1234567bpmn")
    # Apache AGE (PostgreSQL)
    AGE_DSN: str = os.getenv(
        "AGE_DSN",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    ).strip()
    AGE_GRAPH_NAME: str = os.getenv("AGE_GRAPH_NAME", "pdf2bpmn").strip()
    
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
    
    # ------------------------------------------------------------------
    # Semantic dedup (현재 실행 task/role 끼리 임베딩 cosine + 휴리스틱 교차검증)
    # ------------------------------------------------------------------
    # Task/Role 정규화 시 임베딩 사용 (false 면 휴리스틱만 사용)
    ENABLE_SEMANTIC_DEDUP: bool = (os.getenv("ENABLE_SEMANTIC_DEDUP", "true").lower() == "true")
    # task 임베딩 cosine 임계
    # - 강한 휴리스틱 (substring 포함 또는 verb 교집합 + noun Jaccard >= TASK_NOUN_JACCARD_MIN) 통과 시
    #   TASK_SEMANTIC_COSINE_MIN 만 충족하면 merge
    # - 그 외 (cosine 단독 신호) TASK_SEMANTIC_HIGH_COSINE 요구
    TASK_SEMANTIC_COSINE_MIN: float = float(os.getenv("TASK_SEMANTIC_COSINE_MIN", "0.85"))
    TASK_SEMANTIC_HIGH_COSINE: float = float(os.getenv("TASK_SEMANTIC_HIGH_COSINE", "0.92"))
    TASK_NOUN_JACCARD_MIN: float = float(os.getenv("TASK_NOUN_JACCARD_MIN", "0.6"))
    # role 임베딩 cosine 임계 (이 이상 + (display key 일치 OR task signature 충분) 시 merge)
    ROLE_SEMANTIC_COSINE_MIN: float = float(os.getenv("ROLE_SEMANTIC_COSINE_MIN", "0.92"))

    # Performance optimization options
    EVIDENCE_MODE: str = os.getenv("EVIDENCE_MODE", "full")  # "full", "reference_only", "off"
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "fixed")  # "fixed", "semantic"
    # Temporary bypass switch for local tests:
    # when true, force EXTRACT LLM input sections to exactly one section.
    FORCE_SINGLE_SECTION: bool = os.getenv("FORCE_SINGLE_SECTION", "false").lower() == "true"

    # Skill/agent post-processing policy
    # 정책: "유사 지침 2개 이상이면 스킬, 동일 역할자가 같은 스킬 2회 이상이면 에이전트".
    # - SKILL_EXTRACTION_MIN_RATIO 는 0.0 으로 두어 activity 수에 따라 threshold 가 동적으로
    #   올라가는 부작용을 제거하고 min_count 만 사용.
    # - AGENT_CREATION_REQUIRE_AUTOMATION 는 false 가 기본. 자동화 키워드(자동/검증/...)가 없는
    #   업무라도 동일 스킬을 반복 수행하는 lane 이라면 에이전트 후보로 잡는다.
    SKILL_EXTRACTION_MIN_RATIO: float = float(os.getenv("SKILL_EXTRACTION_MIN_RATIO", "0.0"))
    SKILL_EXTRACTION_MIN_COUNT: int = int(os.getenv("SKILL_EXTRACTION_MIN_COUNT", "2"))
    AGENT_CREATION_MIN_TASKS_PER_SKILL_PER_LANE: int = int(
        os.getenv("AGENT_CREATION_MIN_TASKS_PER_SKILL_PER_LANE", "2")
    )
    # 기본값 false: 휴먼 업무라도 동일 스킬을 반복하면 에이전트 후보로 잡는다.
    AGENT_CREATION_REQUIRE_AUTOMATION: bool = (
        os.getenv("AGENT_CREATION_REQUIRE_AUTOMATION", "false").lower() == "true"
    )

    # Skill LLM enrichment
    # - 클러스터링으로 도출한 "공통 지침"을 LLM 으로 풍부한 SOP/스킬 카드로 정제한다.
    # - 실패 시 캐노니컬 문장 기반 폴백을 사용한다.
    SKILL_LLM_ENRICHMENT: bool = (
        os.getenv("SKILL_LLM_ENRICHMENT", "true").lower() == "true"
    )
    SKILL_LLM_MODEL: str = (os.getenv("SKILL_LLM_MODEL") or LLM_MODEL).strip()
    SKILL_LLM_TIMEOUT_SEC: float = float(os.getenv("SKILL_LLM_TIMEOUT_SEC", "60"))
    SKILL_LLM_MAX_CONCURRENCY: int = int(os.getenv("SKILL_LLM_MAX_CONCURRENCY", "3"))
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure output directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories
Config.ensure_dirs()




