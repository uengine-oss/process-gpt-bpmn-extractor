#!/bin/bash
set -e

echo "=============================================="
echo "🚀 PDF2BPMN Container Starting..."
echo "=============================================="

# Print configuration (hide sensitive values)
echo "📋 Configuration:"
echo "   - Agent Server Port: ${AGENT_PORT:-8000}"
echo "   - Neo4j URI: ${NEO4J_URI:-not set}"
echo "   - Supabase URL: ${SUPABASE_URL:+set}"
echo "   - Agent Orchestrator: ${AGENT_ORCH:-pdf2bpmn}"
echo "   - Task Timeout: ${TASK_TIMEOUT:-3600}s"
echo "=============================================="

# If Supabase is running on the host and env points to localhost/127.0.0.1,
# that won't be reachable from inside the container. Rewrite to host.docker.internal.
if [ -n "$SUPABASE_URL" ]; then
    if echo "$SUPABASE_URL" | grep -Eq '^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?(/.*)?$'; then
        OLD_SUPABASE_URL="$SUPABASE_URL"
        export SUPABASE_URL="$(echo "$SUPABASE_URL" | sed -E 's#^https?://(localhost|127\.0\.0\.1)#http://host.docker.internal#')"
        echo "📝 SUPABASE_URL rewritten for Docker: ${OLD_SUPABASE_URL} -> ${SUPABASE_URL}"
    fi
fi

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
AGENT_PORT=${AGENT_PORT:-8000}

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
    
    echo "👋 Graceful shutdown complete!"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT

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
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ All services started successfully!"
echo "   - Agent Server: http://0.0.0.0:${AGENT_PORT}"
echo "=============================================="
echo ""
echo "📡 Container is running. Press Ctrl+C to stop."
echo ""

# Wait for agent process to exit
wait $AGENT_PID

echo "⚠️  Agent server exited unexpectedly!"
cleanup
