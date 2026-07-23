"""A2A server implementation for PDF2BPMN agent."""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from .protocol import (
    DiscoverResponse,
    ExecuteRequest,
    ExecuteResponse,
    TaskStatus,
    TaskEvent,
    TaskEventType,
    TaskResult,
    DEFAULT_INPUT_SCHEMA,
    DEFAULT_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)


class A2AServer:
    """A2A protocol server for PDF2BPMN agent."""
    
    def __init__(
        self,
        agent_executor,
        agent_id: str = "pdf2bpmn",
        agent_name: str = "PDF to BPMN Converter",
        port: int = 9999,
        host: str = "0.0.0.0"
    ):
        """
        Initialize A2A server.
        
        Args:
            agent_executor: PDF2BPMNAgentExecutor instance
            agent_id: Unique agent identifier
            agent_name: Agent display name
            port: Server port (default: 9999)
            host: Server host (default: 0.0.0.0)
        """
        self.agent_executor = agent_executor
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.port = port
        self.host = host
        
        # Task storage (in-memory, can be replaced with Redis/DB)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_events: Dict[str, list] = {}  # task_id -> list of events
        
        # FastAPI app
        self.app = FastAPI(title=f"{agent_name} A2A Server")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/discover", response_model=DiscoverResponse)
        async def discover():
            """Agent discovery endpoint."""
            return DiscoverResponse(
                agent_id=self.agent_id,
                name=self.agent_name,
                description="Extracts BPMN process definitions from PDF documents",
                capabilities=["pdf_analysis", "bpmn_extraction", "process_discovery"],
                input_schema=DEFAULT_INPUT_SCHEMA,
                output_schema=DEFAULT_OUTPUT_SCHEMA,
                version="1.0.0"
            )
        
        @self.app.post("/execute", response_model=ExecuteResponse)
        async def execute(request: ExecuteRequest, background_tasks: BackgroundTasks):
            """Execute a task."""
            task_id = request.task_id or str(uuid.uuid4())
            
            # Validate input
            if not request.input.get("pdf_url") and not request.input.get("pdf_path"):
                raise HTTPException(
                    status_code=400,
                    detail="Either 'pdf_url' or 'pdf_path' must be provided"
                )
            
            # Initialize task
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "input": request.input,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "result": None,
                "error": None
            }
            self.task_events[task_id] = []
            
            # Start background task
            background_tasks.add_task(self._execute_task, task_id, request.input)
            
            return ExecuteResponse(
                task_id=task_id,
                status=TaskStatus.PENDING,
                message="Task queued for execution"
            )
        
        @self.app.get("/status/{task_id}")
        async def get_status(task_id: str):
            """Get task status."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            return {
                "task_id": task_id,
                "status": task["status"],
                "created_at": task["created_at"],
                "result": task.get("result"),
                "error": task.get("error")
            }
        
        @self.app.get("/result/{task_id}", response_model=TaskResult)
        async def get_result(task_id: str):
            """Get task result."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            
            if task["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task is not completed yet. Current status: {task['status']}"
                )
            
            # Collect artifacts from events
            artifacts = []
            for event in self.task_events.get(task_id, []):
                if event["event_type"] == TaskEventType.ARTIFACT:
                    artifacts.append(event["data"])
            
            return TaskResult(
                task_id=task_id,
                status=task["status"],
                result=task.get("result"),
                error=task.get("error"),
                artifacts=artifacts
            )
        
        @self.app.get("/events/{task_id}")
        async def stream_events(task_id: str):
            """Stream task events via Server-Sent Events."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            async def event_generator():
                """Generate SSE events."""
                # Send existing events
                for event in self.task_events.get(task_id, []):
                    yield {
                        "event": event["event_type"],
                        "data": event["data"]
                    }
                
                # Watch for new events
                last_count = len(self.task_events.get(task_id, []))
                task = self.tasks[task_id]
                
                while task["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    current_count = len(self.task_events.get(task_id, []))
                    
                    if current_count > last_count:
                        # New events available
                        for event in self.task_events[task_id][last_count:]:
                            yield {
                                "event": event["event_type"],
                                "data": event["data"]
                            }
                        last_count = current_count
                    
                    await asyncio.sleep(0.5)
                    task = self.tasks[task_id]  # Refresh task status
                
                # Send final event
                if task["status"] == TaskStatus.COMPLETED:
                    yield {
                        "event": TaskEventType.COMPLETE,
                        "data": {"status": "completed", "result": task.get("result")}
                    }
                elif task["status"] == TaskStatus.FAILED:
                    yield {
                        "event": TaskEventType.ERROR,
                        "data": {"status": "failed", "error": task.get("error")}
                    }
            
            return EventSourceResponse(event_generator())
        
        @self.app.delete("/task/{task_id}")
        async def cancel_task(task_id: str):
            """Cancel a task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel task with status: {task['status']}"
                )
            
            task["status"] = TaskStatus.CANCELLED
            self._add_event(task_id, TaskEventType.ERROR, {
                "status": "cancelled",
                "message": "Task cancelled by user"
            })
            
            return {"task_id": task_id, "status": TaskStatus.CANCELLED}
    
    def _add_event(self, task_id: str, event_type: TaskEventType, data: Dict[str, Any]):
        """Add an event to the task event stream."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if task_id not in self.task_events:
            self.task_events[task_id] = []
        self.task_events[task_id].append(event)
        logger.debug(f"[A2A] Task {task_id} event: {event_type} - {data}")
    
    async def _execute_task(self, task_id: str, input_data: Dict[str, Any]):
        """Execute the task using agent executor."""
        task = self.tasks[task_id]
        
        try:
            # Update status
            task["status"] = TaskStatus.RUNNING
            self._add_event(task_id, TaskEventType.PROGRESS, {
                "status": "running",
                "progress": 0,
                "message": "Starting PDF analysis..."
            })
            
            # Create a mock RequestContext and EventQueue for the executor
            # Try to import A2A SDK, fallback to mock if not available
            try:
                from a2a.server.agent_execution import RequestContext, EventQueue
                SDK_AVAILABLE = True
            except ImportError:
                # Fallback: use mock classes
                SDK_AVAILABLE = False
                logger.warning("A2A SDK not available, using mock classes")
            
            class MockRequestContext:
                def __init__(self, input_data, task_id):
                    self.input_data = input_data
                    self.task_id = task_id
                
                def get_user_input(self) -> str:
                    # Extract user input from input_data
                    pdf_url = input_data.get("pdf_url")
                    pdf_path = input_data.get("pdf_path")
                    pdf_file_name = input_data.get("pdf_file_name", "document.pdf")
                    
                    if pdf_url:
                        return f"[InputData] pdf_file_url: {pdf_url}, pdf_file_name: {pdf_file_name}"
                    elif pdf_path:
                        # Convert local path to file:// URL if needed
                        abs_path = Path(pdf_path).absolute()
                        return f"[InputData] pdf_file_url: file://{abs_path}, pdf_file_name: {pdf_file_name}"
                    return ""
                
                def get_context_data(self) -> Dict[str, Any]:
                    # PDF2BPMNAgentExecutor expects a 'row' with specific fields
                    return {
                        "row": {
                            "id": self.task_id,
                            "task_id": self.task_id,
                            "root_proc_inst_id": self.task_id,
                            "proc_inst_id": self.task_id,
                            "tenant_id": input_data.get("tenant_id") or "",
                            "query": self.get_user_input(),
                            "description": f"PDF to BPMN conversion: {input_data.get('pdf_file_name', 'document.pdf')}"
                        },
                        "query": self.get_user_input(),
                        **input_data
                    }
            
            class MockEventQueue:
                def __init__(self, server, task_id):
                    self.server = server
                    self.task_id = task_id
                
                def enqueue_event(self, event):
                    # Convert A2A SDK events to A2A protocol events
                    if hasattr(event, 'message') and hasattr(event, 'status'):
                        # TaskStatusUpdateEvent
                        progress = getattr(event, 'progress', 0)
                        message = getattr(event, 'message', '')
                        self.server._add_event(self.task_id, TaskEventType.PROGRESS, {
                            "status": "running",
                            "progress": progress,
                            "message": message
                        })
                    elif hasattr(event, 'artifact'):
                        # TaskArtifactUpdateEvent
                        artifact = getattr(event, 'artifact', {})
                        self.server._add_event(self.task_id, TaskEventType.ARTIFACT, {
                            "type": "bpmn",
                            "artifact": artifact
                        })
            
            context = MockRequestContext(input_data, task_id)
            event_queue = MockEventQueue(self, task_id)
            
            # Execute using agent executor
            await self.agent_executor.execute(context, event_queue)
            
            # Task completed successfully
            # Note: The executor should have sent completion events
            # For now, we'll mark as completed
            task["status"] = TaskStatus.COMPLETED
            task["result"] = {
                "status": "completed",
                "message": "BPMN extraction completed successfully"
            }
            
            self._add_event(task_id, TaskEventType.COMPLETE, {
                "status": "completed",
                "result": task["result"]
            })
            
        except Exception as e:
            logger.error(f"[A2A] Task {task_id} failed: {e}", exc_info=True)
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)
            self._add_event(task_id, TaskEventType.ERROR, {
                "status": "failed",
                "error": str(e)
            })
    
    def run(self):
        """Run the A2A server."""
        logger.info(f"[A2A] Starting server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
