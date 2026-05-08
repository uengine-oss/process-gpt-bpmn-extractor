"""
General-purpose file → PDF conversion utilities.

Goals:
- Support non-PDF uploads (docx/xlsx/pptx/images/...) by converting them to a PDF for downstream parsing.
- Be flexible: try best available converter and gracefully fallback with actionable errors.

Preferred conversion engines:
1) LibreOffice (soffice) headless conversion for Office docs.
2) Pillow for image → single-page PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import os
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from ..config import Config


OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".odt",
    ".odp",
    ".ods",
    ".rtf",
    ".hwp",
    ".hwpx",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class FileToPdfError(RuntimeError):
    pass


def _pick_available_pdf_path(out_dir: Path, stem: str) -> Path:
    """
    Choose a PDF output path under out_dir that won't overwrite existing files.
    """
    base = "".join(ch for ch in (stem or "converted") if ch not in r'\/:*?"<>|').strip() or "converted"
    candidate = out_dir / f"{base}.pdf"
    if not candidate.exists():
        return candidate
    for _ in range(50):
        suffix = uuid.uuid4().hex[:8]
        candidate = out_dir / f"{base}_{suffix}.pdf"
        if not candidate.exists():
            return candidate
    # Extremely unlikely unless directory is polluted
    return out_dir / f"{base}_{uuid.uuid4().hex}.pdf"


def _find_soffice() -> Optional[str]:
    if Config.LIBREOFFICE_PATH:
        p = Path(Config.LIBREOFFICE_PATH)
        if p.exists():
            return str(p)
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found

    # Windows fallback: common LibreOffice install locations
    windows_candidates = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.com"),
    ]
    for c in windows_candidates:
        if c.exists():
            return str(c)
    return None


def convert_to_pdf(input_path: str, output_dir: str) -> str:
    """
    Convert input file to PDF and return resulting PDF path.
    If input is already a PDF, returns the original path.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileToPdfError(f"입력 파일이 존재하지 않습니다: {src}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".pdf":
        return str(src)

    if not Config.ENABLE_FILE_CONVERSION:
        raise FileToPdfError(f"PDF가 아닌 파일 업로드는 비활성화되어 있습니다: {src.name}")

    ext = src.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return _image_to_pdf(src, output_root)

    if ext in OFFICE_EXTENSIONS:
        return _office_to_pdf(src, output_root)

    # Unknown types: try LibreOffice anyway (it can handle many formats)
    return _office_to_pdf(src, output_root, allow_unknown=True)


def _image_to_pdf(src: Path, out_dir: Path) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise FileToPdfError(
            "이미지 파일을 PDF로 변환하려면 Pillow가 필요합니다. "
            "pip install pillow 또는 이미지 파일 대신 PDF로 업로드하세요."
        ) from e

    out_path = _pick_available_pdf_path(out_dir, src.stem)
    img = Image.open(src)
    # ensure RGB (Pillow requires RGB for PDF in many cases)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.save(out_path, "PDF", resolution=Config.OCR_DPI)
    return str(out_path)


def _office_to_pdf(src: Path, out_dir: Path, allow_unknown: bool = False) -> str:
    # OOXML files are zip-based; fail early with clearer message when extension/content mismatch
    if src.suffix.lower() in {".xlsx", ".docx", ".pptx"} and not zipfile.is_zipfile(src):
        raise FileToPdfError(
            f"파일 확장자({src.suffix})와 실제 파일 포맷이 일치하지 않습니다: {src.name}. "
            "원본 파일이 손상되었거나 다른 형식 파일이 잘못된 확장자로 업로드되었을 수 있습니다."
        )

    soffice = _find_soffice()
    if not soffice:
        raise FileToPdfError(
            "Office 문서를 PDF로 변환하려면 LibreOffice(soffice)가 필요합니다. "
            "서버/컨테이너에 LibreOffice를 설치하거나, PDF로 변환 후 업로드하세요."
        )

    # Use a per-conversion isolated directory to avoid:
    # - picking up stale PDFs from previous runs
    # - collisions between concurrent conversions
    work_dir = out_dir / f".convert_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run soffice in headless mode
    # NOTE: Use --convert-to pdf:writer_pdf_Export for docs sometimes, but generic 'pdf' is okay.
    cmd = [
        soffice,
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--convert-to",
        "pdf",
        "--outdir",
        str(work_dir),
        str(src),
    ]
    # LibreOffice can fail on Windows when inherited Python env points elsewhere.
    # Isolate process environment to avoid "Could not find platform independent libraries".
    run_env = dict(os.environ)
    run_env.pop("PYTHONHOME", None)
    run_env.pop("PYTHONPATH", None)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            env=run_env,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise FileToPdfError(f"LibreOffice 변환이 타임아웃되었습니다: {src.name}") from e

    if proc.returncode != 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        if allow_unknown:
            raise FileToPdfError(
                f"파일을 PDF로 변환하지 못했습니다: {src.name}\n"
                f"stdout={proc.stdout[-2000:]}\n"
                f"stderr={proc.stderr[-2000:]}"
            )
        raise FileToPdfError(
            f"Office 파일을 PDF로 변환하지 못했습니다: {src.name}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )

    try:
        candidates = sorted(work_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileToPdfError(f"PDF 변환 결과를 찾지 못했습니다: work_dir={work_dir}")

        produced = candidates[0]
        final_out = _pick_available_pdf_path(out_dir, src.stem)
        shutil.move(str(produced), str(final_out))
        return str(final_out)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

