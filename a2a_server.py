#!/usr/bin/env python3
"""
A2A Server for PDF2BPMN Agent
Google A2A 프로토콜을 준수하는 서버 구현
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 환경 변수 로드
env_path = current_dir / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Loaded env from: {env_path}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PDF2BPMN AgentExecutor import
from pdf2bpmn_agent_executor import PDF2BPMNAgentExecutor
from src.pdf2bpmn.a2a.server import A2AServer


def main():
    """Main entry point for A2A server."""
    # PDF2BPMN AgentExecutor 설정
    executor_config = {
        "pdf2bpmn_url": os.getenv("PDF2BPMN_URL", "http://localhost:8001"),
        "timeout": int(os.getenv("TASK_TIMEOUT", "3600")),
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_key": os.getenv("SERVICE_ROLE_KEY")
    }
    
    # AgentExecutor 생성
    executor = PDF2BPMNAgentExecutor(config=executor_config)
    logger.info("PDF2BPMNAgentExecutor initialized")
    
    # A2A 서버 설정
    server_port = int(os.getenv("A2A_SERVER_PORT", "9999"))
    server_host = os.getenv("A2A_SERVER_HOST", "0.0.0.0")
    
    # A2A 서버 생성 및 실행
    server = A2AServer(
        agent_executor=executor,
        agent_id="pdf2bpmn",
        agent_name="PDF to BPMN Converter",
        port=server_port,
        host=server_host
    )
    
    print()
    print("=" * 70)
    print("[*] A2A Server for PDF2BPMN Agent")
    print("=" * 70)
    print(f"[>] Server URL: http://{server_host}:{server_port}")
    print(f"[>] Agent ID: pdf2bpmn")
    print(f"[>] Agent Name: PDF to BPMN Converter")
    print()
    print("[*] Endpoints:")
    print(f"  - GET  http://{server_host}:{server_port}/discover")
    print(f"  - POST http://{server_host}:{server_port}/execute")
    print(f"  - GET  http://{server_host}:{server_port}/status/{{task_id}}")
    print(f"  - GET  http://{server_host}:{server_port}/result/{{task_id}}")
    print(f"  - GET  http://{server_host}:{server_port}/events/{{task_id}}")
    print()
    print("[!] Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
