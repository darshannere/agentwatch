from __future__ import annotations


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AgentWatch server")
    parser.add_argument("command", choices=["serve"], help="Command to run")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db", default="agentwatch.db")
    args = parser.parse_args()

    if args.command == "serve":
        import os

        os.environ["AGENTWATCH_DB"] = args.db

        import uvicorn

        uvicorn.run("agentwatch.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
