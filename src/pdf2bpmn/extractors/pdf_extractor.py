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

    # 사용자 [도구 설정] 의 pdf2bpmnLevel 과 동일한 값 (concise / standard / detailed).
    # SOP 분할 강도 (= LLM 이 SOP 경계를 얼마나 적극적으로 잡을지) 를 조절한다.
    # 절대 하드코딩된 N 개 분할을 강제하지 않는다 — 단일 절차 문서는 모든 레벨에서 1 개 SOP 가 정답.
    _ALLOWED_SEGMENTATION_LEVELS = {"concise", "standard", "detailed"}

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, chunking_strategy: str = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.chunking_strategy = chunking_strategy or Config.CHUNKING_STRATEGY
        self._segmentation_level: str = "standard"

    def set_segmentation_level(self, level: str | None) -> None:
        """SOP 분할 강도 (concise / standard / detailed) 적용. 잘못된 값은 standard 로 폴백."""
        normalized = (level or "standard").strip().lower()
        if normalized not in self._ALLOWED_SEGMENTATION_LEVELS:
            normalized = "standard"
        self._segmentation_level = normalized
    
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
        Heading-based fallback is only used when SOP segmentation produced
        nothing usable. We never override an LLM SOP result merely because it
        returned a single section — a single SOP is a valid (and often correct)
        outcome for a document that describes one process end-to-end.

        - sop_sections empty   → use heading split as fallback (if available).
        - sop_sections present → trust SOP result (including the single-SOP case).
        """
        if sop_sections:
            return False
        return bool(heading_sections)

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

        # ── 사용자 도구 설정 (pdf2bpmnLevel) 에 따른 분할 강도 ──
        # 하드코딩된 SOP 개수 강제는 절대 없다. 단일 절차 문서는 모든 레벨에서 1 개 SOP.
        # LLM 의 분할 _임계_ 만 system role 메시지로 강하게 박아 일관성을 높인다.
        level = (getattr(self, "_segmentation_level", None) or "standard").strip().lower()
        if level == "concise":
            level_label = "간결"
            system_msg = (
                "당신은 문서를 **큰 단위로 묶는** 분석 전문가입니다.\n"
                "사용자가 [간결] 모드를 선택했습니다. 당신의 임무는 다음과 같습니다:\n"
                "\n"
                "1) 결과 SOP 수를 적게 유지하세요. 같은 큰 업무 영역에 속하는 장/조/별표는 한 SOP 로 묶습니다.\n"
                "2) **명백히 다른 업무 도메인** 일 때만 SOP 를 분리하세요. 예: 신청 vs 점검 vs 사후관리, "
                "   서로 다른 신청 종류, 서로 다른 제도/조직.\n"
                "3) 별표/항/세부 절차/표는 가능한 부모 SOP 안의 task 로 흡수합니다.\n"
                "4) 망설여지면 **항상 묶는 쪽** 을 선택하세요.\n"
                "5) 단일 절차 문서면 1 개 SOP 가 정답입니다.\n"
                "\n"
                "이 임무를 절대 잊지 마세요. 다음 사용자 메시지에 분할 기준 상세와 문서가 주어집니다.\n"
                "결과는 항상 [간결] 모드 임무와 일치해야 합니다."
            )
        elif level == "detailed":
            level_label = "상세"
            system_msg = (
                "당신은 문서 구조를 **세밀하게 분리하는** 분석 전문가입니다.\n"
                "사용자가 [상세] 모드를 선택했습니다. 당신의 임무는 다음과 같습니다:\n"
                "\n"
                "1) 명시적 구조 표지 (제N장, 제N조, 별표 N, 표 N, [붙임 N], 1./1.1/가./□ 등) 가 보이면 "
                "   **각 항목을 가능한 별도 SOP 로 분리** 하세요. 망설이지 마세요.\n"
                "2) 같은 업무 영역이라도 절차 단위가 구분되면 (다른 입력/출력 문서, 다른 판단 게이트, "
                "   다른 트리거) **반드시 분리** 합니다.\n"
                "3) 별표/항/세부 절차/표도 독립 SOP 후보로 **적극 검토** 하세요.\n"
                "4) 망설여지면 **항상 분리하는 쪽** 을 선택하세요. 결과 SOP 수가 많아도 괜찮습니다 — "
                "   사용자가 그렇게 요청했습니다.\n"
                "5) **유일한 금지선** — 한 절차의 단순 단계 (접수 → 검토 → 승인 → 통보) 는 분리하지 않습니다. "
                "   단계는 SOP 안의 task 입니다.\n"
                "6) 단일 절차 문서면 [상세] 모드라도 1 개 SOP 가 정답입니다 — 단, 정말 구조 표지가 하나도 "
                "   없을 때만.\n"
                "\n"
                "이 임무를 절대 잊지 마세요. 다음 사용자 메시지에 분할 기준 상세와 문서가 주어집니다.\n"
                "결과는 항상 [상세] 모드 임무와 일치해야 합니다 — 구조 표지가 다수면 결과 SOP 도 다수여야 합니다."
            )
        else:
            level_label = "표준"
            system_msg = (
                "당신은 문서 구조를 따라 **합리적으로 분할하는** 분석 전문가입니다.\n"
                "사용자가 [표준] 모드를 선택했습니다. 당신의 임무는 다음과 같습니다:\n"
                "\n"
                "** 가장 중요한 원칙 **:\n"
                "- 문서가 _단순하고 짧은 단일 절차_ 가 아니라면 **거의 항상 SOP 를 2 개 이상으로 분할** 하세요.\n"
                "- 분할에 의심이 들면 **분할하는 쪽을 선택** 하세요. 1 개로 묶는 것은 정말 단일 절차일 때만.\n"
                "- 페이지가 여러 장이고 단계가 명확히 구분되어 있다면 분할이 거의 확실히 정답입니다.\n"
                "\n"
                "** 분할 기준 **:\n"
                "1) 명시적 장/절/조 (제N장 / 제N조 / 제N편 / 별표 N) 가 **2 개 이상** 보이면 → 그 단위로 분할.\n"
                "2) 큰 업무 단계 (접수 / 검토 / 승인 / 통보 / 사후관리) 가 명확히 구분되면 → 그 단위로 분할.\n"
                "3) 입력/출력 문서가 명확히 다른 업무로 전환되면 → 분할.\n"
                "4) 별표/항/세부 표는 부모 SOP 의 task 로 흡수 (이 점만 [상세] 와 다름).\n"
                "\n"
                "** 1 개 SOP 가 정답인 경우 (드물게만 해당) **:\n"
                "- 문서가 정말 단일 절차이고 구조 표지가 하나도 없을 때.\n"
                "- 예: 휴가 신청 1 페이지 문서, 단순한 업무 메모.\n"
                "\n"
                "이 임무를 절대 잊지 마세요. 다음 사용자 메시지에 분할 기준 상세와 문서가 주어집니다.\n"
                "결과는 항상 [표준] 모드 임무와 일치해야 합니다 — 구조가 있는 문서면 분할이 정답입니다."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", """[현재 모드: """ + level_label + """]
당신은 위 system 메시지의 임무를 절대 잊으면 안 됩니다. 출력 직전 한 번 더 확인하세요.

[기본 원칙]
- 분할 여부는 **문서 자체의 구조** 가 결정합니다. 강제 최소 개수도 없고, 강제 최대 개수도 없습니다.
- 문서가 단일 절차만 설명하면 1개 SOP 가 정답.
- 문서가 여러 절차/장/조/별표/표를 명시적 구조로 담고 있으면 그 구조 그대로 N개 SOP 가 정답.
- 페이지 수, 청크 수, 토큰 수 같은 형식적 수치 자체로는 분할하지 마세요.
  단, **문서가 직접 보여주는 구조 표지** (제N장, 제N조, 별표 N, 표 N, 1./1.1/가./□ 같은 번호 체계,
  업무 항목 목록, 절차 목록 표) 는 강한 분할 신호로 간주하세요.

[새 SOP 시작의 강한 신호 — 하나라도 보이면 분할 우선]
A) **문서 구조 표지의 변화** (가장 강한 신호):
   - "제1장 / 제2장", "제1조 / 제2조", "별표 1 / 별표 2", "1. / 2.", "1.1 / 1.2", "[붙임 1] / [붙임 2]"
     같은 번호/제목 체계가 여러 개 등장하고, 각 항목이 독립적인 업무/규정/절차를 설명함.
   - 문서 앞부분의 목차/조항 목록/업무 목록 표가 N개 항목을 나열하고, 본문이 그 구조대로 전개됨.
   - 절차 흐름도/업무 분장표/RACI 표 등이 여러 개 등장.
2) 업무 목적이 바뀜 (예: "자료 접수 절차" → "사후 점검 절차").
3) 주 수행 역할/부서가 바뀜.
4) 주 입력 문서 또는 주 출력 문서가 바뀜.
5) 절차의 트리거(시작 조건)와 종료 조건이 독립적임 (한 절차의 분기가 아니라 서로 다른 입력으로 시작).

A가 보이면 거의 항상 분할이 정답입니다. A가 없을 때 2~5를 검토하세요.

[새 SOP 로 나누지 말아야 하는 경우 — 같은 SOP 안에 둘 것]
- 같은 절차의 연속 세부 단계 (예: 접수 → 1차 검토 → 2차 검토 → 승인 → 통보 가 한 절차의 흐름이면 1개 SOP).
  단계마다 SOP 를 만들지 마세요. 단계는 SOP 안의 task 입니다.
- 같은 역할이 같은 목적으로 이어서 수행하는 작업.
- 한 절차 안의 분기/조건 (반려/승인 분기는 분할 사유 아님).
- 단순 설명 보강, 예시, 주의사항, 체크리스트.
- 같은 절차를 다른 말로 반복 설명한 부분.
- 표/그림 캡션 같은 보조 자료.

[1 개와 N 개, 둘 다 정답이 될 수 있습니다]
- "신청 → 검토 → 승인 → 통보" 한 절차만 설명하는 문서 → 1 개 SOP.
- "제1장 / 제2장 / 제3장 ... 제N장" 처럼 N 개의 독립 절차/규정/업무를 다루는 절차서 → N 개 SOP.
- 두려워하지 마세요. 문서가 정말 그렇게 구성돼 있으면 20개도 30개도 정답입니다.
  반대로 정말 단일 절차면 1 개도 정답입니다.

[과분할 금지선]
- 한 절차의 세부 단계마다 SOP 만들지 말 것.
- 사소한 표현/문체 차이로 새 SOP 만들지 말 것.
- 페이지가 바뀌었다는 이유만으로 경계 만들지 말 것 (단, 새 페이지가 새 장/조 시작이면 OK).

[과소분할 금지선]
- 문서에 명시적 장/조/별표/표 번호가 여러 개 등장하는데도 1 개로 묶지 말 것.
- 서로 다른 업무를 한 SOP 안에 묶어 놓지 말 것.
- "전체가 결국 같은 영역의 업무" 라는 이유로 합치지 말 것 — 영역이 같아도 절차 단위는 다를 수 있음.

[출력 시 자기 점검 — 반드시 통과시킬 것]
1) **현재 모드 ([""" + level_label + """]) 의 임무와 결과가 일치하는가?**
   - [간결] 모드인데 SOP 가 많이 나왔다면 → 같은 영역끼리 더 묶어 줄여야 합니다.
   - [상세] 모드인데 SOP 가 적게 나왔고 문서에 구조 표지가 다수라면 → 더 분리해야 합니다.
   - [표준] 모드인데 결과가 극단적 (1 개 또는 30 개 이상) 이라면 → 강한 구조 표지 기준으로 재검토.
2) 문서에 "제N장 / 제N조 / 별표 N / N. / [붙임 N]" 같은 번호 체계가 보이는가?
   보인다면 그 번호 체계를 모드에 맞게 반영했는가?
3) 분할 근거가 A 또는 2~5 중 무엇인지 스스로 답할 수 있는가? 답할 수 없으면 분할하지 마세요.
4) 한 절차의 단순 단계마다 SOP 만들고 있지 않은가? (모든 모드에서 금지)
5) [간결] 모드에서 구조 표지를 무리하게 따라 너무 잘게 쪼개고 있지 않은가?
6) [상세] 모드에서 명시적 구조 표지를 보고도 "한 영역 업무니까 합치자" 라고 판단하고 있지 않은가?

[page_from / page_to]
- 실제 절차/조항이 시작/끝나는 페이지를 기록. 겹침 없이.

[응답 형식(JSON)]
{{"sops":[{{"title":"SOP 제목","page_from":1,"page_to":3}}]}}

[예시 1 — 1 개 SOP 가 정답]
- 문서: 한 절차의 단계 (신청 접수 → 서류 검토 → 승인 → 결과 통보) 만 설명.
- 단계마다 분기/반복이 있어도 절차는 하나.
- 정답: 1 개 SOP (예: "휴가 신청 및 검증 절차"). 단계마다 SOP 4 개로 쪼개지 않음.

[예시 2 — 여러 SOP 가 정답 (장/조 구조)]
- 문서: "제1장 총칙", "제2장 신청", "제3장 심의", "제4장 결과 통보", "제5장 사후관리", "별표 1 ~ N"
  같이 명시적 장/조/별표 번호로 나뉜 절차서.
- 정답: 각 장/조/별표 단위로 SOP 다수 (10 개 ~ 30 개도 정상).

[예시 3 — 표/목차 기반 다수 SOP]
- 문서 앞에 "업무 목록 표" 가 있고 본문이 그 표의 항목 순서대로 각 업무를 설명.
- 정답: 표의 각 행을 1 개 SOP 로 분리.

[예시 4 — 헷갈리지만 1 개로 두는 경우]
- 문서: "신청 → 검토 → 승인 → 통보" 가 길게 풀려 있고 각 단계가 자세히 서술됨.
- 명시적 장/조 번호 없음, 같은 절차의 단계.
- 정답: 1 개 SOP.

문서 내용:
{text}
""")
        ])
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



