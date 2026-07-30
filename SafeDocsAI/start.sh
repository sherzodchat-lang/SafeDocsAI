#!/bin/bash

# AndozAI Quick Start Script

echo "🚀 Starting AndozAI..."
echo

POSTGRES_HOST="${POSTGRES_SERVER:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://localhost:11434}"
OLLAMA_MODEL_CHAT="${OLLAMA_MODEL_CHAT:-gemma3n:e4b}"
OLLAMA_MODEL_EMBEDDING="${OLLAMA_MODEL_EMBEDDING:-nomic-embed-text}"

check_postgres() {
    if command -v pg_isready >/dev/null 2>&1; then
        pg_isready -q -h "$POSTGRES_HOST" -p "$POSTGRES_PORT"
        return $?
    fi

    if command -v nc >/dev/null 2>&1; then
        nc -z "$POSTGRES_HOST" "$POSTGRES_PORT" >/dev/null 2>&1
        return $?
    fi

    # Fallback when pg_isready/nc are unavailable.
    (echo >"/dev/tcp/$POSTGRES_HOST/$POSTGRES_PORT") >/dev/null 2>&1
}

# Check if postgres is running
if ! check_postgres; then
    echo "❌ PostgreSQL is not reachable."
    echo "🔄 Attempting to start/resume Docker services..."

    # Start/Run PostgreSQL
    if docker ps -a --format '{{.Names}}' | grep -q "^andozai-postgres$"; then
        echo "   Starting existing andozai-postgres container..."
        docker start andozai-postgres >/dev/null
    else
        echo "   Creating and starting new andozai-postgres container..."
        docker run --name andozai-postgres \
            -e POSTGRES_PASSWORD=andozai_password \
            -e POSTGRES_USER=andozai_user \
            -e POSTGRES_DB=andozai_db \
            -p 5432:5432 \
            -v postgres_data:/var/lib/postgresql/data \
            -d postgres:15.2-alpine >/dev/null
    fi

    # Start/Run ChromaDB (check independently or just ensure it is running)
    if docker ps -a --format '{{.Names}}' | grep -q "^andozai-chromadb$"; then
         if ! docker ps --format '{{.Names}}' | grep -q "^andozai-chromadb$"; then
            echo "   Starting existing andozai-chromadb container..."
            docker start andozai-chromadb >/dev/null
         fi
    else
        echo "   Creating and starting new andozai-chromadb container..."
        docker run --name andozai-chromadb \
            -p 8000:8000 \
            -v chroma_data:/chroma/chroma \
            -e IS_PERSISTENT=TRUE \
            -e PERSIST_DIRECTORY=/chroma/chroma \
            -d chromadb/chroma:latest >/dev/null
    fi

    echo "⏳ Waiting for services to initialize..."
    sleep 5
    
    # Re-check
    if ! check_postgres; then
         echo "❌ Failed to start PostgreSQL. Please check docker logs."
         exit 1
    fi
    echo "✅ Services started successfully."
fi

echo "✅ PostgreSQL is running ($POSTGRES_HOST:$POSTGRES_PORT)"

# Check if ollama is running
if ! curl -fsS "$OLLAMA_API_BASE/api/tags" >/dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Please start it:"
    echo "   ollama serve"
else
    echo "✅ Ollama is running ($OLLAMA_API_BASE)"

    if command -v ollama >/dev/null 2>&1; then
        ensure_ollama_model() {
            local model="$1"

            if [ -z "$model" ]; then
                return 0
            fi

            if ! OLLAMA_HOST="$OLLAMA_API_BASE" ollama list | grep -Fq "$model"; then
                echo "⬇️  Pulling Ollama model: $model"
                OLLAMA_HOST="$OLLAMA_API_BASE" ollama pull "$model"
            fi
        }

        ensure_ollama_model "$OLLAMA_MODEL_CHAT"
        ensure_ollama_model "$OLLAMA_MODEL_EMBEDDING"
    fi
fi

# Start backend
echo "🐍 Starting Backend..."
cd backend
source venv/bin/activate
# Initialize DB
echo "📦 Initializing Database..."
python -m app.init_db >/dev/null 2>&1
python run.py --reload &
BACKEND_PID=$!
cd ..

sleep 2

# Start frontend
echo "⚛️  Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo "═══════════════════════════════════════════════════════════"
echo "  AndozAI is starting up..."
echo "═══════════════════════════════════════════════════════════"
echo
echo "  Backend:  http://localhost:8001"
echo "  Frontend: http://localhost:5173"
echo "  API Docs: http://localhost:8001/api/v1/docs"
echo
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════════════════════"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
