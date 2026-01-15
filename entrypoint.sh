#!/bin/bash
set -e

echo "=============================================="
echo "🚀 PDF2BPMN Container Starting..."
echo "=============================================="

# Print configuration (hide sensitive values)
echo "📋 Configuration:"
echo "   - API Server Port: ${API_PORT:-8001}"
echo "   - Agent Server Port: ${AGENT_PORT:-8000}"
echo "   - PDF2BPMN URL: ${PDF2BPMN_URL:-http://localhost:8001}"
echo "   - Neo4j URI: ${NEO4J_URI:-not set}"
echo "   - Supabase URL: ${SUPABASE_URL:+set}"
echo "   - Agent Orchestrator: ${AGENT_ORCH:-pdf2bpmn}"
echo "   - Task Timeout: ${TASK_TIMEOUT:-3600}s"
echo "=============================================="

# Wait for dependencies if needed
if [ -n "$WAIT_FOR_NEO4J" ]; then
    echo "⏳ Waiting for Neo4j at ${NEO4J_URI}..."
    for i in $(seq 1 30); do
        if python -c "from src.pdf2bpmn.graph.neo4j_client import Neo4jClient; c = Neo4jClient(); c.verify_connection(); c.close()" 2>/dev/null; then
            echo "✅ Neo4j is ready"
            break
        fi
        echo "   Attempt $i/30..."
        sleep 2
    done
fi

# Set default ports
API_PORT=${API_PORT:-8001}
AGENT_PORT=${AGENT_PORT:-8000}

# Update PDF2BPMN_URL if not explicitly set
if [ -z "$PDF2BPMN_URL" ]; then
    export PDF2BPMN_URL="http://localhost:${API_PORT}"
    echo "📝 PDF2BPMN_URL set to: $PDF2BPMN_URL"
fi

# Function to handle graceful shutdown
# Kubernetes sends SIGTERM and waits for terminationGracePeriodSeconds (3600s)
cleanup() {
    echo ""
    echo "🛑 Received shutdown signal..."
    echo "⏳ Waiting for current tasks to complete (this may take up to 1 hour)..."
    
    # Send SIGTERM to agent server first (allow it to finish current task)
    kill -SIGTERM $AGENT_PID 2>/dev/null || true
    
    # Wait for agent server to finish gracefully
    # It will complete current PDF processing before exiting
    echo "⏳ Waiting for agent server to complete current work..."
    wait $AGENT_PID 2>/dev/null || true
    echo "✅ Agent server stopped"
    
    # Then stop API server
    kill -SIGTERM $API_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
    echo "✅ API server stopped"
    
    echo "👋 Graceful shutdown complete!"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT

# Start FastAPI server (PDF2BPMN API) in background
echo ""
echo "🔧 Starting PDF2BPMN API Server on port ${API_PORT}..."
python run.py api --host 0.0.0.0 --port ${API_PORT} &
API_PID=$!

# Wait for API server to be ready
echo "⏳ Waiting for API server to be ready..."
sleep 5

# Check if API server started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo "❌ API server failed to start!"
    exit 1
fi

# Verify API server is responding
for i in $(seq 1 10); do
    if curl -s http://localhost:${API_PORT}/api/health > /dev/null 2>&1; then
        echo "✅ API server is ready"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "⚠️  API server health check failed, but continuing..."
    fi
    sleep 2
done

# Start Agent Server (ProcessGPT SDK) in background
echo ""
echo "🤖 Starting PDF2BPMN Agent Server on port ${AGENT_PORT}..."
python pdf2bpmn_agent_server.py &
AGENT_PID=$!

# Wait a bit for agent server to initialize
sleep 3

# Check if agent server started successfully
if ! kill -0 $AGENT_PID 2>/dev/null; then
    echo "❌ Agent server failed to start!"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ All services started successfully!"
echo "   - API Server: http://0.0.0.0:${API_PORT}"
echo "   - API Docs: http://0.0.0.0:${API_PORT}/docs"
echo "   - Agent Server: http://0.0.0.0:${AGENT_PORT}"
echo "=============================================="
echo ""
echo "📡 Container is running. Press Ctrl+C to stop."
echo ""

# Wait for any process to exit
wait -n $API_PID $AGENT_PID

# If one process exits, stop the other
echo "⚠️  One of the services exited unexpectedly!"
cleanup
