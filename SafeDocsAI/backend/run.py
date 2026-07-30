#!/usr/bin/env python3
"""
Development server runner for AndozAI backend.
For production, use: uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run AndozAI backend server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1; reverse proxy fronts it)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                   AndozAI Backend                        ║
╠══════════════════════════════════════════════════════════╣
║  Server: http://{args.host}:{args.port:<5}                      ║
║  Docs:   http://{args.host}:{args.port}/api/v1/docs              ║
║  Health: http://{args.host}:{args.port}/health                   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
