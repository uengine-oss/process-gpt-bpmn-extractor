#!/usr/bin/env python3
"""
PDF2BPMN Agent Server
ProcessGPT SDK를 사용한 PDF to BPMN 변환 에이전트 서버
"""

import asyncio
import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# .env.local 파일 로드
try:
    from dotenv import load_dotenv
    # .env.local 먼저 시도, 없으면 .env
    env_local = current_dir / '.env.local'
    env_file = current_dir / '.env'
    agent_env = current_dir / 'agent.env'
    
    if env_local.exists():
        load_dotenv(env_local)
        print(f"[OK] Loaded env from: {env_local}")
    elif agent_env.exists():
        load_dotenv(agent_env)
        print(f"[OK] Loaded env from: {agent_env}")
    elif env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded env from: {env_file}")
    else:
        print("[WARN] .env.local or .env file not found.")
except ImportError:
    print("[WARN] python-dotenv not installed. Using system env vars.")

# ProcessGPT SDK imports
try:
    from processgpt_agent_sdk import ProcessGPTAgentServer
    PROCESSGPT_SDK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ProcessGPT SDK not available: {e}")
    print("pip install processgpt-agent-sdk 로 설치하세요.")
    PROCESSGPT_SDK_AVAILABLE = False

# 로컬 모듈 imports
from pdf2bpmn_agent_executor import PDF2BPMNAgentExecutor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDF2BPMNServerConfig:
    """PDF2BPMN 에이전트 서버 설정"""
    
    def __init__(self):
        # ProcessGPT 설정
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.service_role_key = os.getenv("SERVICE_ROLE_KEY")
        self.polling_interval = int(os.getenv("POLLING_INTERVAL", "5"))
        self.agent_orch = os.getenv("AGENT_ORCH", "pdf2bpmn")
        
        # PDF2BPMN 서버 설정
        self.pdf2bpmn_url = os.getenv("PDF2BPMN_URL", "http://localhost:8001")
        self.task_timeout = int(os.getenv("TASK_TIMEOUT", "300"))  # 5분
        
        # 환경 검증
        self.validate()
    
    def validate(self):
        """설정 검증"""
        missing_vars = []
        
        if not self.supabase_url:
            missing_vars.append("SUPABASE_URL")
        if not self.supabase_anon_key:
            missing_vars.append("SUPABASE_ANON_KEY")
            
        if missing_vars:
            raise ValueError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return {
            "supabase_url": self.supabase_url,
            "polling_interval": self.polling_interval,
            "agent_orch": self.agent_orch,
            "pdf2bpmn_url": self.pdf2bpmn_url,
            "task_timeout": self.task_timeout
        }


class PDF2BPMNServerManager:
    """PDF2BPMN 에이전트 서버 관리자"""
    
    def __init__(self, config: PDF2BPMNServerConfig):
        self.config = config
        self.server: ProcessGPTAgentServer = None
        self.executor: PDF2BPMNAgentExecutor = None
        self.is_running = False
        
        # 신호 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """신호 핸들러 - 우아한 종료"""
        logger.info(f"신호 {signum} 수신 - 서버 종료 중...")
        if self.server:
            self.server.stop()
        self.is_running = False
    
    async def initialize(self):
        """서버 초기화"""
        logger.info("PDF2BPMN 에이전트 서버 초기화 중...")
        
        # AgentExecutor 설정
        executor_config = {
            "pdf2bpmn_url": self.config.pdf2bpmn_url,
            "timeout": self.config.task_timeout,
            "supabase_url": self.config.supabase_url,
            "supabase_key": self.config.service_role_key
        }
        
        # AgentExecutor 생성
        self.executor = PDF2BPMNAgentExecutor(config=executor_config)
        logger.info(f"PDF2BPMNAgentExecutor 생성됨 (Server: {self.config.pdf2bpmn_url})")
        
        if not PROCESSGPT_SDK_AVAILABLE:
            logger.error("ProcessGPT SDK가 설치되지 않았습니다.")
            return False
        
        # ProcessGPT 서버 생성
        try:
            self.server = ProcessGPTAgentServer(
                agent_executor=self.executor,
                agent_type=self.config.agent_orch
            )
            self.server.polling_interval = self.config.polling_interval
            logger.info(f"ProcessGPT 서버 생성됨 - 에이전트: {self.config.agent_orch}")
            return True
            
        except Exception as e:
            logger.error(f"ProcessGPT 서버 생성 실패: {e}")
            return False
    
    async def start(self):
        """서버 시작"""
        if not await self.initialize():
            logger.error("서버 초기화 실패")
            return False
        
        self.is_running = True
        
        print()
        print("=" * 70)
        print("[*] PDF2BPMN Agent Server")
        print("=" * 70)
        print(f"[>] Start Time: {datetime.now().isoformat()}")
        print(f"[>] Agent Type: {self.config.agent_orch}")
        print(f"[>] Polling Interval: {self.config.polling_interval}s")
        print(f"[>] PDF2BPMN Server: {self.config.pdf2bpmn_url}")
        print(f"[>] Task Timeout: {self.config.task_timeout}s")
        print()
        print("[*] Supported Tasks:")
        print("  - PDF to BPMN conversion")
        print("  - Multi-process extraction from single PDF")
        print("  - Auto-save to proc_def and proc_map")
        print("  - Real-time progress events")
        print()
        print("[*] Query Example:")
        print('  \'{"pdf_url": "https://xxx.supabase.co/storage/.../file.pdf"}\'')
        print('  \'[InputData] pdf_file_url: https://xxx/file.pdf, pdf_file_name: manual.pdf\'')
        print()
        print("[!] Press Ctrl+C to stop the server")
        print("=" * 70)
        print()
        
        try:
            # ProcessGPT 서버 실행 (무한 폴링 루프)
            await self.server.run()
            
        except KeyboardInterrupt:
            logger.info("사용자가 서버 중지를 요청했습니다")
        except Exception as e:
            logger.error(f"서버 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
    
    async def stop(self):
        """서버 중지"""
        logger.info("서버 종료 중...")
        self.is_running = False
        
        if self.server:
            try:
                self.server.stop()
                logger.info("ProcessGPT 서버 중지됨")
            except Exception as e:
                logger.warning(f"서버 중지 중 오류: {e}")
        
        print("\n[OK] PDF2BPMN Agent Server stopped gracefully")


def print_usage():
    """사용법 출력"""
    print()
    print("=" * 60)
    print("PDF2BPMN Agent Server - PDF to BPMN 변환 에이전트")
    print("=" * 60)
    print()
    print("필수 환경변수:")
    print("  SUPABASE_URL          - Supabase 프로젝트 URL")
    print("  SUPABASE_ANON_KEY     - Supabase 익명 키")
    print()
    print("선택적 환경변수:")
    print("  SERVICE_ROLE_KEY      - Supabase 서비스 역할 키 (DB 저장용)")
    print("  PDF2BPMN_URL=http://localhost:8001 - PDF2BPMN 서버 URL")
    print("  POLLING_INTERVAL=5    - 폴링 간격 (초)")
    print("  AGENT_ORCH=pdf2bpmn   - 에이전트 타입")
    print("  TASK_TIMEOUT=300      - 태스크 타임아웃 (초)")
    print()
    print("실행 예시:")
    print("  export SUPABASE_URL='https://your-project.supabase.co'")
    print("  export SUPABASE_ANON_KEY='your-anon-key'")
    print("  python pdf2bpmn_agent_server.py")
    print()
    print("또는 agent.env 파일에 환경변수를 설정하세요.")
    print()


async def main():
    """메인 실행 함수"""
    try:
        # 설정 로드 및 검증
        config = PDF2BPMNServerConfig()
        
        # 서버 관리자 생성 및 시작
        server_manager = PDF2BPMNServerManager(config)
        await server_manager.start()
        
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        print_usage()
        sys.exit(1)
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Python 버전 체크
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher required")
        sys.exit(1)
    
    # 필수 패키지 체크
    missing_packages = []
    
    if not PROCESSGPT_SDK_AVAILABLE:
        missing_packages.append("processgpt-agent-sdk")
    
    try:
        import httpx
    except ImportError:
        missing_packages.append("httpx")
    
    try:
        from supabase import create_client
    except ImportError:
        missing_packages.append("supabase")
    
    if missing_packages:
        print(f"[ERROR] Missing required packages: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        sys.exit(1)
    
    # 메인 실행
    asyncio.run(main())
