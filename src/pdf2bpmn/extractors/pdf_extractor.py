"""PDF text and structure extraction."""
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Generator

import pdfplumber
import requests

from ..models.entities import Document, Section, ReferenceChunk, generate_id
from ..config import Config


def _llm_timeout_sec() -> float:
    try:
        v = float(os.getenv("OPENAI_TIMEOUT_SEC", "120"))
    except Exception:
        v = 120.0
    return max(1.0, min(v, 120.0))


class PDFExtractor:
    """Extract text and structure from PDF files."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, chunking_strategy: str = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.chunking_strategy = chunking_strategy or Config.CHUNKING_STRATEGY
    
    def extract_document(self, pdf_path: str) -> tuple[Document, list[Section], list[ReferenceChunk]]:
        """Extract document structure and content from PDF."""
        path = Path(pdf_path)
        
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            synap_page_texts: dict[int, str] = {}
            if Config.ENABLE_OCR and (Config.OCR_ENGINE or "").lower() == "synap":
                synap_page_texts = self._extract_text_with_synap(path, page_count=page_count)
            
            # Create document
            doc = Document(
                doc_id=generate_id(),
                title=path.stem,
                source=str(path),
                page_count=page_count
            )
            
            # Extract all text with page info
            all_text = []
            page_texts = {}
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                synap_text = synap_page_texts.get(i + 1, "")
                if synap_text:
                    text = _merge_text_lines(text, synap_text)

                # If the page contains images, always attempt OCR/Vision extraction (configurable).
                # This is NOT a fallback only when text is empty; images can contain critical tables/steps.
                if (
                    Config.ENABLE_OCR
                    and i < Config.OCR_MAX_PAGES
                    and not synap_text
                ):
                    try:
                        has_images = bool(getattr(page, "images", None)) and len(page.images) > 0
                    except Exception:
                        has_images = False

                    if has_images and Config.OCR_ALWAYS_IF_IMAGES:
                        ocr_text = self._extract_text_from_page_image(path, page_index=i)
                        if ocr_text:
                            # Merge without exploding duplicates: append unique lines only.
                            merged = _merge_text_lines(text, ocr_text)
                            text = merged
                page_texts[i + 1] = text
                all_text.append((i + 1, text))

            # Extract sections
            # 1) SOP boundary detection (optional, LLM-based)
            # 2) Heading-based split
            # 3) Choose strategy with a guard against over-collapsed single-SOP outputs
            sections = []
            sop_sections: list[Section] = []
            heading_sections: list[Section] = []
            if Config.ENABLE_SOP_SEGMENTATION and Config.OPENAI_API_KEY:
                try:
                    sop_sections = self._extract_sop_sections(doc.doc_id, page_texts)
                except Exception as e:
                    print(f"[WARN] SOP segmentation failed, fallback to heading split: {e}")
                    sop_sections = []

            heading_sections = self._extract_sections(doc.doc_id, all_text)

            if self._should_use_heading_sections(
                sop_sections=sop_sections,
                heading_sections=heading_sections,
                page_count=page_count,
            ):
                sections = heading_sections
                print(
                    f"[SECTION-STRATEGY] using heading split "
                    f"(sop={len(sop_sections)}, heading={len(heading_sections)}, pages={page_count})"
                )
            else:
                sections = sop_sections or heading_sections
                print(
                    f"[SECTION-STRATEGY] using sop split "
                    f"(sop={len(sop_sections)}, heading={len(heading_sections)}, pages={page_count})"
                )

            if Config.FORCE_SINGLE_SECTION:
                sections = self._force_single_section(doc.doc_id, page_texts)
            
            # Create reference chunks
            if self.chunking_strategy == "semantic":
                chunks = self._create_semantic_chunks(doc.doc_id, page_texts)
            else:
                chunks = self._create_chunks(doc.doc_id, page_texts)
            
            return doc, sections, chunks

    def _extract_text_with_synap(self, pdf_path: Path, page_count: int) -> dict[int, str]:
        """Extract OCR text for a full document via Synap DocuAnalyzer."""
        base_url = (Config.SYNAP_OCR_BASE_URL or "").rstrip("/")
        api_key = (Config.SYNAP_OCR_API_KEY or "").strip()
        if not base_url or not api_key:
            print("[WARN] Synap OCR is enabled but SYNAP_OCR_BASE_URL or SYNAP_OCR_API_KEY is missing.")
            return {}

        timeout_sec = max(5.0, float(Config.SYNAP_OCR_TIMEOUT_SEC))
        poll_interval_sec = max(0.2, float(Config.SYNAP_OCR_POLL_INTERVAL_SEC))
        max_pages = max(0, min(int(Config.OCR_MAX_PAGES), int(page_count)))
        if max_pages == 0:
            return {}

        fid: str | None = None
        try:
            with pdf_path.open("rb") as fh:
                upload_response = requests.post(
                    f"{base_url}/da",
                    data={
                        "api_key": api_key,
                        "type": "upload",
                    },
                    files={
                        "file": (
                            pdf_path.name,
                            fh,
                            "application/octet-stream",
                        )
                    },
                    timeout=timeout_sec,
                )
            upload_response.raise_for_status()
            upload_body = upload_response.json()
            fid = str(((upload_body.get("result") or {}).get("fid") or "")).strip()
            if not fid:
                raise ValueError(f"Synap upload response missing fid: {upload_body}")

            status_result = self._wait_for_synap_completion(
                base_url=base_url,
                api_key=api_key,
                fid=fid,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
            )
            total_pages = int((status_result or {}).get("total_pages") or page_count or 0)
            result_pages = min(max_pages, total_pages or max_pages)
            texts: dict[int, str] = {}

            for page_index in range(1, result_pages + 1):
                page_response = requests.post(
                    f"{base_url}/result/{fid}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "api_key": api_key,
                        "page_index": page_index,
                        # Keep Synap output aligned with the legacy OCR path:
                        # downstream expects plain OCR text with preserved line breaks.
                        "type": "text",
                    },
                    timeout=timeout_sec,
                )
                page_response.raise_for_status()
                page_text = (page_response.text or "").strip()
                if page_text:
                    texts[page_index] = page_text
            return texts
        except Exception as exc:
            print(f"[WARN] Synap OCR failed, falling back to local OCR path: {exc}")
            return {}
        finally:
            if fid:
                try:
                    requests.post(
                        f"{base_url}/delete/{fid}",
                        json={"api_key": api_key},
                        timeout=timeout_sec,
                    )
                except Exception:
                    pass

    def _wait_for_synap_completion(
        self,
        *,
        base_url: str,
        api_key: str,
        fid: str,
        timeout_sec: float,
        poll_interval_sec: float,
    ) -> dict:
        """Poll Synap until the uploaded document reaches SUCCESS status."""
        deadline = time.time() + timeout_sec
        last_payload: dict = {}

        while time.time() < deadline:
            response = requests.post(
                f"{base_url}/filestatus/{fid}",
                json={"api_key": api_key},
                timeout=timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            last_payload = payload
            result = payload.get("result") or {}
            status = str(result.get("filestatus") or "").upper()

            if status == "SUCCESS":
                return result
            if status and status not in {"RUNNING", "QUEUED", "PENDING"}:
                raise RuntimeError(f"Synap OCR failed with status={status}: {payload}")

            time.sleep(poll_interval_sec)

        raise TimeoutError(f"Timed out waiting for Synap OCR completion: {last_payload}")

    def _should_use_heading_sections(
        self,
        *,
        sop_sections: list[Section],
        heading_sections: list[Section],
        page_count: int,
    ) -> bool:
        """
        Choose heading-based sections when SOP segmentation looks over-collapsed.
        This keeps behavior stable across model/provider differences.
        """
        if not heading_sections:
            return False
        if not sop_sections:
            return True

        # If SOP gives only one broad section while heading split finds multiple,
        # prefer heading split for stability.
        if len(sop_sections) == 1 and len(heading_sections) >= 2:
            s = sop_sections[0]
            spans_almost_all = (int(s.page_from) <= 1) and (int(s.page_to) >= max(1, page_count - 1))
            if spans_almost_all or page_count >= 6:
                return True
        return False

    def _force_single_section(self, doc_id: str, page_texts: dict[int, str]) -> list[Section]:
        """Force all pages into exactly one section (temporary test bypass)."""
        pages = sorted(page_texts.keys())
        if not pages:
            return []
        combined = "\n\n".join((page_texts.get(p) or "").strip() for p in pages).strip()
        if not combined:
            return []
        return [
            Section(
                section_id=generate_id(),
                doc_id=doc_id,
                heading="Forced Single Section",
                level=1,
                page_from=pages[0],
                page_to=pages[-1],
                content=combined,
            )
        ]

    def _extract_sop_sections(self, doc_id: str, page_texts: dict[int, str]) -> list[Section]:
        """
        문서 전체에서 SOP(독립 프로세스) 경계를 LLM으로 식별하고,
        SOP 단위로 Section을 생성합니다.
        """
        # Limit pages for boundary detection
        pages = sorted(page_texts.keys())[: Config.SOP_MAX_PAGES_FOR_BOUNDARY]
        joined = "\n\n".join([f"[PAGE {p}]\n{page_texts.get(p, '')}" for p in pages])
        if not joined.strip():
            return []

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore
            from langchain_core.output_parsers import JsonOutputParser  # type: ignore
            from pydantic import BaseModel, Field  # type: ignore
        except Exception:
            return []

        class _SOPBoundary(BaseModel):
            title: str = Field(...)
            page_from: int = Field(...)
            page_to: int = Field(...)

        class _SOPBoundaries(BaseModel):
            sops: list[_SOPBoundary] = Field(default_factory=list)

        prompt = ChatPromptTemplate.from_template(
            """당신은 문서 구조 분석 전문가입니다.
가장 중요한 목표:
- 목표는 문서를 최대한 많이 자르는 것이 아니라, **업무적으로 의미 있는 적절한 단위로 안정적으로 분할하는 것**입니다.
- 아래에 정의된 **명시적 규칙만** 사용해 SOP 경계를 판단하세요.
- 이 규칙에 없는 이유로 SOP를 새로 만들거나 없애지 마세요.

주어진 문서에서 SOP(표준작업절차) 또는 독립적인 업무 프로세스의 경계를 식별해주세요.

규칙:
- 문서 제목 전체는 SOP가 아닐 수 있습니다. 문서 내부의 구체적인 업무 절차(요청관리, 변경관리, 장애관리 등)를 찾으세요.
- SOP는 독립적인 시작/끝을 가진 절차 단위입니다.
- 반환은 최대한 포괄적으로 하되, 과도하게 잘게 쪼개지 마세요.
- 문서에 절/장/단계/프로세스 단위가 여러 개 보이면 SOP를 여러 개로 나누세요.
- 서로 다른 목적/트리거/종료조건을 가진 절차는 반드시 별도 SOP로 분리하세요.
- 전체 문서를 1개 SOP로 묶는 것은 정말 단일 절차 문서일 때만 허용됩니다.
- 문서가 여러 페이지이거나 여러 장/절/단계/업무 항목을 포함하면, 특별한 사유가 없는 한 **최소 3개 이상의 SOP**를 우선 검토하세요.
- 다만 과도한 세분화도 금지합니다. 한두 문단/사소한 문장 차이만으로 SOP를 새로 만들지 마세요.
- SOP 하나는 어느 정도 독립적인 업무 목적/절차 흐름을 가져야 하며, 지나치게 잘게 쪼개어 20개, 30개, 40개 이상으로 불필요하게 분할하지 마세요.
- 문서 구조를 존중하되, **비슷한 목적의 연속 단계**는 하나의 SOP로 묶으세요.
- 경계가 보이면 분리하고, 비슷한 목적의 연속 단계는 묶어 **업무적으로 의미 있는 적절한 단위**로 분할하세요.
- 섹션 개수를 늘리는 것보다, **명시적인 업무 경계와 문서 구조를 기준으로 안정적으로 분할**하는 것을 우선하세요.
- 사소한 표현 차이, 문장 길이 차이, 설명 문체 차이 때문에 경계를 바꾸지 마세요.
- 경계는 문서의 명시적 구조와 업무 흐름 변화에 근거해 결정하세요.

- SOP 경계 판단 규칙(반드시 아래 순서대로 적용):
  1) 문서에 명시적인 장/절/단계 제목 변화가 있으면, 새 SOP 시작을 우선 검토하세요.
  2) 업무 목적이 바뀌면 새 SOP를 시작하세요.
  3) 주요 담당 역할/부서가 바뀌면 새 SOP를 시작하세요.
  4) 주요 입력 문서 또는 출력 문서가 바뀌면 새 SOP를 시작하세요.
  5) 승인/판단/검토 게이트가 새로 등장하면 새 SOP를 시작하세요.
  6) 준비 단계에서 본처리 단계로 전환되면 새 SOP를 시작하세요.
  7) 처리 단계에서 결과 통보/사후관리 단계로 전환되면 새 SOP를 시작하세요.
  8) 위 조건이 없으면 새 SOP를 만들지 말고 기존 SOP에 포함하세요.

- 새 SOP를 시작해야 하는 대표 조건(체크리스트):
  - 업무 목적이 바뀜
  - 주요 담당 역할/부서가 바뀜
  - 주요 입력 문서나 출력 문서가 바뀜
  - 승인/판단/검토 게이트가 새로 등장함
  - 준비 단계에서 본처리 단계로 넘어감
  - 처리 단계에서 결과 통보/사후관리 단계로 넘어감
  - 앞 단계와 독립적으로 설명 가능한 새로운 절차 묶음이 시작됨

- 새 SOP로 나누지 말아야 하는 대표 조건(체크리스트):
  - 같은 목적의 연속 세부 단계
  - 같은 역할이 이어서 수행하는 세부 작업 나열
  - 단순 설명 보강, 예시, 주의사항
  - 같은 절차 안의 하위 체크리스트
  - 앞 절차의 세부 실행 방법만 더 자세히 풀어쓴 부분
  - 같은 절차를 다른 말로 반복 설명한 부분
  - 같은 입력과 같은 목적을 가진 연속 업무 설명

- 1개 SOP만 반환하는 것은 다음 경우에만 허용됩니다:
  1) 문서 길이가 매우 짧고
  2) 처음부터 끝까지 하나의 단일 절차만 설명하며
  3) 준비/본처리/후속처리를 나눌 정도의 구조 변화가 전혀 없을 때
- 문서가 6페이지 이상이거나, 제목/번호 체계가 여러 개 보이거나, 단계가 3개 이상 보이면 1개 SOP로 반환하지 마세요.
- 한 SOP 안에 너무 많은 단계가 몰려 있으면 다음 기준으로 쪼개세요:
  - 접수/준비 단계
  - 검토/판단 단계
  - 승인/처리 단계
  - 결과 통보/사후관리 단계
- 제목이 다르거나, 담당 역할이 크게 바뀌거나, 입력/출력 문서가 바뀌거나, 판단/승인 게이트가 새로 등장하면 새 SOP 시작으로 보세요.
- page_from/page_to는 실제 경계를 반영해 겹침 없이 작성하세요.
- 비슷한 하위 절차가 연속으로 반복될 때는 개별 단계마다 SOP를 만들지 말고 상위 절차 단위로 통합하세요.
- 각 SOP 경계는 왜 분리되는지 내부적으로 판단하세요.
- 경계 이유 없이 페이지를 기계적으로 자르지 마세요.
- 한 문서 안에서는 가능한 한 **동일한 granularity(분할 세기)**를 유지하세요.
- 앞부분은 크게 묶고 뒷부분은 잘게 쪼개는 식으로 불균형하게 분할하지 마세요.
- 같은 유형의 절차 묶음이 반복되면 비슷한 수준으로 묶으세요.
- 한 SOP는 최소한 다음 중 하나를 설명해야 합니다:
  - 하나의 독립적인 업무 목적
  - 하나의 독립적인 검토/판단 단계
  - 하나의 독립적인 승인/처리 단계
  - 하나의 독립적인 통보/사후관리 단계
- 위 기준 없이 단순히 문장이 바뀌었다는 이유만으로 SOP를 새로 만들지 마세요.

응답 형식(JSON):
{{"sops":[{{"title":"SOP 제목","page_from":1,"page_to":3}}]}}

예시 1:
- 문서 구성: 신청 접수 -> 서류 검토 -> 승인 심사 -> 결과 통보
- 올바른 출력: 최소 3~4개 SOP
  - 신청 접수
  - 서류 검토
  - 승인 심사
  - 결과 통보
- 잘못된 출력: 문서 전체를 1개 SOP로 묶음

예시 2:
- 문서에 "제1장 접수", "제2장 심사", "제3장 사후관리"가 보임
- 올바른 출력:
  - 접수 절차 (page_from/page_to)
  - 심사 절차 (page_from/page_to)
  - 사후관리 절차 (page_from/page_to)

예시 3:
- 같은 업무 문서라도 준비/처리/종료 단계가 뚜렷하면 별도 SOP로 분리
- 예:
  - 개최 준비
  - 본 심의 진행
  - 결과 통보 및 후속조치

좋은 예시:
- 신청 접수 / 서류 검토 / 승인 심사 / 결과 통보
- 접수 절차 / 심사 절차 / 사후관리 절차

나쁜 예시:
- 신청서 열람 / 신청서 확인 / 신청서 저장 / 신청서 표시 를 각각 별도 SOP로 분리
- 문서 전체를 무조건 1개 SOP로 묶기
- 페이지 수만 보고 기계적으로 페이지당 1개 SOP로 자르기

최종 판단 원칙:
- 출력 전 반드시 스스로 점검하세요:
  - 전체 문서를 1개로 과도하게 묶지 않았는가?
  - 반대로 사소한 단계 차이만으로 과도하게 쪼개지 않았는가?
  - 각 SOP는 독립적인 업무 목적과 흐름을 가지는가?
  - 경계 판단을 제목/목적/역할/입출력/게이트 변화 같은 명시적 기준에 근거해 수행했는가?
- 1개로 과소분할하지 마세요.
- 40개 이상처럼 과도하게 과분할하지 마세요.
- 같은 문장을 반복 설명한 부분 때문에 SOP 수를 늘리지 마세요.
- 경계 근거가 약하면 새 SOP를 만들기보다 기존 SOP 포함 여부를 먼저 검토하세요.
- 문서 구조가 보이면 그 구조를 적극 반영하세요.
- 확신이 낮을 때는 무조건 합치거나 무조건 쪼개지 말고, 업무적으로 더 자연스러운 단위를 선택하세요.

문서 내용:
{text}
"""
        )
        llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            base_url=(Config.LLM_BASE_URL or None),
            temperature=0,
            timeout=_llm_timeout_sec(),
        )
        parser = JsonOutputParser(pydantic_object=_SOPBoundaries)
        chain = prompt | llm | parser
        data = chain.invoke({"text": joined})
        boundaries = _SOPBoundaries(**data)
        if not boundaries.sops:
            return []

        sections: list[Section] = []
        for sop in boundaries.sops:
            pf = max(1, int(sop.page_from))
            pt = max(pf, int(sop.page_to))
            content = "\n\n".join([page_texts.get(p, "") for p in range(pf, pt + 1)]).strip()
            if not content:
                continue
            sections.append(
                Section(
                    section_id=generate_id(),
                    doc_id=doc_id,
                    heading=sop.title.strip() or "SOP",
                    level=1,
                    page_from=pf,
                    page_to=pt,
                    content=content,
                )
            )

        return sections

    def _extract_text_from_page_image(self, pdf_path: Path, page_index: int) -> str:
        """
        Render a PDF page to an image and extract text via OCR/Vision.
        Tries multiple render/OCR backends; always fails gracefully (returns "").
        """
        image = None

        # 1) Render via PyMuPDF if available (no external poppler dependency)
        try:
            import fitz  # type: ignore

            with fitz.open(str(pdf_path)) as doc:
                if page_index < 0 or page_index >= doc.page_count:
                    return ""
                page = doc.load_page(page_index)
                zoom = max(1.0, Config.OCR_DPI / 72.0)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                if pix.width * pix.height > Config.OCR_MAX_IMAGE_PIXELS:
                    # Downscale if too large
                    scale = (Config.OCR_MAX_IMAGE_PIXELS / float(pix.width * pix.height)) ** 0.5
                    mat = fitz.Matrix(zoom * scale, zoom * scale)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                from PIL import Image  # type: ignore

                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception:
            image = None

        # 2) Render via pdfplumber (best-effort)
        if image is None:
            try:
                from PIL import Image  # type: ignore
                import io

                with pdfplumber.open(pdf_path) as pdf:
                    if page_index < 0 or page_index >= len(pdf.pages):
                        return ""
                    page = pdf.pages[page_index]
                    page_image = page.to_image(resolution=Config.OCR_DPI)
                    # page_image.original is a PIL Image in many environments
                    image = getattr(page_image, "original", None)
                    if image is None:
                        # Fallback to bytes if possible
                        bio = io.BytesIO()
                        page_image.save(bio, format="PNG")
                        bio.seek(0)
                        image = Image.open(bio)
            except Exception:
                image = None

        if image is None:
            return ""

        # OCR engine selection
        engine = (Config.OCR_ENGINE or "tesseract").lower()

        if engine == "synap":
            return self._ocr_with_tesseract(image)
        if engine == "openai_vision":
            return self._ocr_with_openai_vision(image)

        # Default: tesseract
        return self._ocr_with_tesseract(image)

    def _ocr_with_tesseract(self, image) -> str:
        try:
            import pytesseract  # type: ignore
        except Exception:
            return ""

        try:
            # Korean+English is common for business docs
            text = pytesseract.image_to_string(image, lang="kor+eng")
            return (text or "").strip()
        except Exception:
            return ""

    def _ocr_with_openai_vision(self, image) -> str:
        """
        OCR via OpenAI Vision (multimodal). This is used when Tesseract isn't available or quality is insufficient.
        """
        try:
            import base64
            import io
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.messages import HumanMessage  # type: ignore
        except Exception:
            return ""

        try:
            buf = io.BytesIO()
            # Keep PNG for better text clarity
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            llm = ChatOpenAI(
                model=Config.OCR_MODEL,
                api_key=Config.OPENAI_API_KEY,
                base_url=(Config.OCR_BASE_URL or None),
                temperature=0,
                timeout=_llm_timeout_sec(),
            )
            prompt = (
                "You are an OCR engine. Extract ALL readable Korean/English text from the image. "
                "Preserve line breaks. Do not add commentary. Output plain text only."
            )
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            )
            resp = llm.invoke([msg])
            text = getattr(resp, "content", "") or ""
            return str(text).strip()
        except Exception:
            return ""


    def _extract_sections(
        self, 
        doc_id: str, 
        page_texts: list[tuple[int, str]]
    ) -> list[Section]:
        """Extract section hierarchy from document."""
        sections = []
        
        # Patterns for detecting headings
        heading_patterns = [
            (1, r'^#{1}\s+(.+)$'),  # Markdown style
            (1, r'^제\s*\d+\s*장\s*(.+)$'),  # Korean chapter
            (2, r'^제\s*\d+\s*절\s*(.+)$'),  # Korean section
            (2, r'^#{2}\s+(.+)$'),
            (1, r'^\d+\.\s+([A-Z가-힣].+)$'),  # Numbered heading
            (2, r'^\d+\.\d+\s+(.+)$'),
            (3, r'^\d+\.\d+\.\d+\s+(.+)$'),
            (1, r'^[IVX]+\.\s+(.+)$'),  # Roman numerals
            (2, r'^[A-Z]\.\s+(.+)$'),  # Letter headings
        ]
        
        current_section = None
        section_start_page = 1
        
        for page_num, text in page_texts:
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for heading patterns
                for level, pattern in heading_patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        # Save previous section
                        if current_section:
                            current_section.page_to = page_num - 1
                            sections.append(current_section)
                        
                        # Create new section
                        current_section = Section(
                            section_id=generate_id(),
                            doc_id=doc_id,
                            heading=line,
                            level=level,
                            page_from=page_num,
                            page_to=page_num,  # Will be updated
                            content=""
                        )
                        section_start_page = page_num
                        break
                
                # Add content to current section
                if current_section:
                    current_section.content += line + "\n"
        
        # Close last section
        if current_section:
            current_section.page_to = page_texts[-1][0] if page_texts else 1
            sections.append(current_section)
        
        # If no sections detected, create one for whole document
        if not sections:
            full_text = "\n".join(text for _, text in page_texts)
            sections.append(Section(
                section_id=generate_id(),
                doc_id=doc_id,
                heading="Document Content",
                level=1,
                page_from=1,
                page_to=len(page_texts),
                content=full_text
            ))
        
        return sections
    
    def _create_chunks(
        self, 
        doc_id: str, 
        page_texts: dict[int, str]
    ) -> list[ReferenceChunk]:
        """Create overlapping text chunks for embedding."""
        chunks = []
        
        for page_num, text in page_texts.items():
            if not text.strip():
                continue
            
            # Split into sentences/paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            chunk_start = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_chunk) + len(para) > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            doc_id, page_num, chunk_start, current_chunk
                        ))
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + para
                    chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(para) - 1
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para
            
            # Save remaining chunk
            if current_chunk:
                chunks.append(self._create_chunk(
                    doc_id, page_num, chunk_start, current_chunk
                ))
        
        return chunks
    
    def _create_semantic_chunks(
        self, 
        doc_id: str, 
        page_texts: dict[int, str]
    ) -> list[ReferenceChunk]:
        """Create semantic chunks based on sections/paragraphs to minimize overlap."""
        chunks = []
        
        # Combine all pages
        full_text = []
        for page_num in sorted(page_texts.keys()):
            text = page_texts[page_num]
            if text.strip():
                full_text.append((page_num, text))
        
        if not full_text:
            return chunks
        
        # Split by major sections (double newlines, headings)
        sections = []
        current_section = []
        current_page = full_text[0][0]
        
        for page_num, text in full_text:
            # Split by paragraphs (double newlines)
            paragraphs = re.split(r'\n\s*\n+', text)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if this is a heading (potential section break)
                is_heading = any(re.match(pattern, para.split('\n')[0], re.MULTILINE) 
                                for pattern in [
                                    r'^제\s*\d+\s*[장절]',
                                    r'^\d+\.\s+[A-Z가-힣]',
                                    r'^#{1,3}\s+',
                                ])
                
                # If heading and current section is large enough, start new section
                if is_heading and current_section and len('\n\n'.join(current_section)) > self.chunk_size * 0.5:
                    sections.append((current_page, '\n\n'.join(current_section)))
                    current_section = [para]
                    current_page = page_num
                else:
                    current_section.append(para)
                    current_page = page_num
        
        # Add last section
        if current_section:
            sections.append((current_page, '\n\n'.join(current_section)))
        
        # Create chunks from sections (minimal overlap)
        for i, (page_num, section_text) in enumerate(sections):
            # If section is too large, split it
            if len(section_text) > self.chunk_size * 1.5:
                # Split large section
                paragraphs = re.split(r'\n\s*\n+', section_text)
                current_chunk = ""
                chunk_start = 0
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                doc_id, page_num, chunk_start, current_chunk
                            ))
                        current_chunk = para
                        chunk_start = 0
                    else:
                        current_chunk += ("\n\n" if current_chunk else "") + para
                
                if current_chunk:
                    chunks.append(self._create_chunk(
                        doc_id, page_num, chunk_start, current_chunk
                    ))
            else:
                # Use entire section as chunk (minimal overlap with previous)
                if i > 0 and self.chunk_overlap > 0:
                    # Add minimal overlap from previous chunk
                    prev_chunk = chunks[-1].text if chunks else ""
                    overlap = prev_chunk[-min(self.chunk_overlap, len(prev_chunk)):] if prev_chunk else ""
                    if overlap:
                        section_text = overlap + "\n\n" + section_text
                
                chunks.append(self._create_chunk(
                    doc_id, page_num, 0, section_text
                ))
        
        return chunks
    
    def _create_chunk(
        self, 
        doc_id: str, 
        page: int, 
        start: int, 
        text: str
    ) -> ReferenceChunk:
        """Create a single reference chunk."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        return ReferenceChunk(
            chunk_id=generate_id(),
            doc_id=doc_id,
            page=page,
            span=f"{start}:{start + len(text)}",
            text=text,
            hash=text_hash
        )
    
    def iter_chunks(self, pdf_path: str) -> Generator[ReferenceChunk, None, None]:
        """Stream chunks from PDF for incremental processing."""
        _, _, chunks = self.extract_document(pdf_path)
        for chunk in chunks:
            yield chunk


def _merge_text_lines(base_text: str, extra_text: str) -> str:
    """
    Merge OCR text into extracted text while minimizing obvious duplication.
    Keeps ordering: base_text first, then any new lines from extra_text.
    """
    base_lines = [ln.strip() for ln in (base_text or "").splitlines() if ln.strip()]
    extra_lines = [ln.strip() for ln in (extra_text or "").splitlines() if ln.strip()]
    if not extra_lines:
        return base_text or ""

    seen = set(base_lines)
    merged = list(base_lines)
    for ln in extra_lines:
        if ln not in seen:
            merged.append(ln)
            seen.add(ln)
    return "\n".join(merged).strip()



