#!/usr/bin/env python3
"""CLI entry point for PDF2BPMN."""
import argparse
import sys
from pathlib import Path

from src.pdf2bpmn.workflow.graph import PDF2BPMNWorkflow
from src.pdf2bpmn.config import Config


def run_cli(pdf_paths: list[str]):
    """Run the PDF to BPMN conversion from CLI."""
    print("=" * 60)
    print("🚀 PDF2BPMN Converter")
    print("=" * 60)
    
    # Validate files
    for path in pdf_paths:
        if not Path(path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {path}")
            return 1
    
    # Initialize workflow
    print("\n📌 워크플로우 초기화 중...")
    workflow = PDF2BPMNWorkflow()
    
    # Initialize Neo4j schema
    print("📊 Neo4j 스키마 초기화 중...")
    try:
        workflow.neo4j.init_schema()
        print("   ✅ Neo4j 연결 성공")
    except Exception as e:
        print(f"   ❌ Neo4j 연결 실패: {e}")
        return 1
    
    # Create initial state
    state = {
        "pdf_paths": pdf_paths,
        "documents": [],
        "sections": [],
        "reference_chunks": [],
        "processes": [],
        "tasks": [],
        "roles": [],
        "gateways": [],
        "events": [],
        "skills": [],
        "dmn_decisions": [],
        "dmn_rules": [],
        "evidences": [],
        "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        "current_step": "ingest_pdf",
        "error": None,
        "bpmn_xml": None,
        "bpmn_xmls": {},
        "bpmn_files": {},
        "skill_docs": {},
        "dmn_xml": None
    }
    
    # Run workflow steps
    try:
        print("\n" + "=" * 60)
        result = workflow.ingest_pdf(state)
        state.update(result)
        
        result = workflow.segment_sections(state)
        state.update(result)
        
        result = workflow.extract_candidates(state)
        state.update(result)
        
        result = workflow.normalize_entities(state)
        state.update(result)
        
        # Continue with generation
        result = workflow.generate_skills(state)
        state.update(result)
        
        result = workflow.generate_dmn(state)
        state.update(result)
        
        result = workflow.assemble_bpmn(state)
        state.update(result)
        
        result = workflow.validate_consistency(state)
        state.update(result)
        
        result = workflow.export_artifacts(state)
        state.update(result)
        
        print("\n" + "=" * 60)
        print("✅ 변환 완료!")
        print("=" * 60)
        print(f"\n📁 출력 위치: {Config.OUTPUT_DIR}")
        print(f"   - process.bpmn")
        if state.get("dmn_xml"):
            print(f"   - decisions.dmn")
        if state.get("skill_docs"):
            print(f"   - {len(state['skill_docs'])}개의 스킬 문서")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        workflow.neo4j.close()


def run_streamlit():
    """Run the Streamlit UI."""
    import subprocess
    app_path = Path(__file__).parent / "src" / "pdf2bpmn" / "ui" / "app.py"
    # Streamlit is an optional extra (see pyproject). Say so plainly instead of
    # letting the subprocess die with a bare "No module named streamlit".
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. It is an optional extra:")
        print("    pip install '.[ui]'      # or:  uv sync --extra ui")
        print("(Left out of the container image: the image never runs the UI,")
        print(" and it pulled 41 critical/high vulnerabilities with it.)")
        return
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    from src.pdf2bpmn.api.main import app
    print(f"🚀 API 서버 시작: http://{host}:{port}")
    print(f"📄 API 문서: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="PDF to BPMN Converter - 업무 편람을 BPMN/DMN/Skill 문서로 변환"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="PDF 파일 변환")
    convert_parser.add_argument(
        "files",
        nargs="+",
        help="변환할 PDF 파일 경로"
    )
    
    # UI command (Streamlit)
    ui_parser = subparsers.add_parser("ui", help="Streamlit UI 실행")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="FastAPI 백엔드 서버 실행")
    api_parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    api_parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Neo4j 스키마 초기화")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        sys.exit(run_cli(args.files))
    elif args.command == "ui":
        run_streamlit()
    elif args.command == "api":
        run_api_server(args.host, args.port)
    elif args.command == "init":
        from src.pdf2bpmn.graph.neo4j_client import Neo4jClient
        client = Neo4jClient()
        try:
            if client.verify_connection():
                print("✅ Neo4j 연결 성공")
                client.init_schema()
                print("✅ 스키마 초기화 완료")
            else:
                print("❌ Neo4j 연결 실패")
        finally:
            client.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

