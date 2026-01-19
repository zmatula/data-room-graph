#!/usr/bin/env python3
"""Setup Neo4j using Docker for the Data Room Graph application."""

import subprocess
import sys
import time
import argparse
from pathlib import Path


# Neo4j Docker configuration
NEO4J_IMAGE = "neo4j:5.26.0-enterprise"
CONTAINER_NAME = "data-room-graph-neo4j"
NEO4J_HTTP_PORT = 7474
NEO4J_BOLT_PORT = 7687
NEO4J_PASSWORD = "dataroomgraph"  # Change in production

# Volume paths (Windows-compatible)
DATA_DIR = Path.home() / ".data-room-graph" / "neo4j" / "data"
LOGS_DIR = Path.home() / ".data-room-graph" / "neo4j" / "logs"
PLUGINS_DIR = Path.home() / ".data-room-graph" / "neo4j" / "plugins"


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command.

    Args:
        cmd: Command and arguments.
        check: Whether to raise on non-zero exit.

    Returns:
        CompletedProcess instance.
    """
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def is_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        result = run_command(["docker", "info"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def container_exists() -> bool:
    """Check if the Neo4j container exists."""
    result = run_command(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        check=False,
    )
    return CONTAINER_NAME in result.stdout


def container_running() -> bool:
    """Check if the Neo4j container is running."""
    result = run_command(
        ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        check=False,
    )
    return CONTAINER_NAME in result.stdout


def create_directories():
    """Create necessary directories for Neo4j data."""
    for dir_path in [DATA_DIR, LOGS_DIR, PLUGINS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def start_neo4j():
    """Start the Neo4j container."""
    if container_running():
        print(f"Container '{CONTAINER_NAME}' is already running.")
        return True

    if container_exists():
        print(f"Starting existing container '{CONTAINER_NAME}'...")
        run_command(["docker", "start", CONTAINER_NAME])
    else:
        print(f"Creating new container '{CONTAINER_NAME}'...")
        create_directories()

        # Convert Windows paths to Docker-compatible paths
        data_mount = str(DATA_DIR).replace("\\", "/")
        logs_mount = str(LOGS_DIR).replace("\\", "/")
        plugins_mount = str(PLUGINS_DIR).replace("\\", "/")

        cmd = [
            "docker", "run",
            "--name", CONTAINER_NAME,
            "-d",
            "-p", f"{NEO4J_HTTP_PORT}:7474",
            "-p", f"{NEO4J_BOLT_PORT}:7687",
            "-v", f"{data_mount}:/data",
            "-v", f"{logs_mount}:/logs",
            "-v", f"{plugins_mount}:/plugins",
            "-e", f"NEO4J_AUTH=neo4j/{NEO4J_PASSWORD}",
            "-e", "NEO4J_ACCEPT_LICENSE_AGREEMENT=yes",
            "-e", "NEO4J_PLUGINS=[\"apoc\"]",
            "-e", "NEO4J_dbms_security_procedures_unrestricted=apoc.*",
            "-e", "NEO4J_dbms_security_procedures_allowlist=apoc.*",
            NEO4J_IMAGE,
        ]
        run_command(cmd)

    # Wait for Neo4j to be ready
    print("Waiting for Neo4j to be ready...")
    for i in range(60):
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", NEO4J_BOLT_PORT))
            sock.close()
            if result == 0:
                print(f"\nNeo4j is ready!")
                print(f"  Browser: http://localhost:{NEO4J_HTTP_PORT}")
                print(f"  Bolt: bolt://localhost:{NEO4J_BOLT_PORT}")
                print(f"  Username: neo4j")
                print(f"  Password: {NEO4J_PASSWORD}")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)

    print("\nWarning: Neo4j may not be fully ready yet.")
    return False


def stop_neo4j():
    """Stop the Neo4j container."""
    if not container_running():
        print(f"Container '{CONTAINER_NAME}' is not running.")
        return

    print(f"Stopping container '{CONTAINER_NAME}'...")
    run_command(["docker", "stop", CONTAINER_NAME])
    print("Container stopped.")


def remove_neo4j():
    """Remove the Neo4j container."""
    if container_running():
        stop_neo4j()

    if not container_exists():
        print(f"Container '{CONTAINER_NAME}' does not exist.")
        return

    print(f"Removing container '{CONTAINER_NAME}'...")
    run_command(["docker", "rm", CONTAINER_NAME])
    print("Container removed.")


def show_status():
    """Show the status of the Neo4j container."""
    if not is_docker_running():
        print("Docker is not running.")
        return

    if container_running():
        print(f"Container '{CONTAINER_NAME}' is RUNNING")
        result = run_command(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "table {{.Status}}"],
            check=False,
        )
        print(result.stdout)
    elif container_exists():
        print(f"Container '{CONTAINER_NAME}' exists but is STOPPED")
    else:
        print(f"Container '{CONTAINER_NAME}' does not exist")


def show_logs(lines: int = 50):
    """Show the Neo4j container logs."""
    if not container_exists():
        print(f"Container '{CONTAINER_NAME}' does not exist.")
        return

    result = run_command(
        ["docker", "logs", "--tail", str(lines), CONTAINER_NAME],
        check=False,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage Neo4j Docker container for Data Room Graph"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "restart", "remove", "status", "logs"],
        help="Command to execute",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of log lines to show (for 'logs' command)",
    )

    args = parser.parse_args()

    if not is_docker_running():
        print("Error: Docker is not running. Please start Docker first.")
        sys.exit(1)

    if args.command == "start":
        start_neo4j()
    elif args.command == "stop":
        stop_neo4j()
    elif args.command == "restart":
        stop_neo4j()
        start_neo4j()
    elif args.command == "remove":
        remove_neo4j()
    elif args.command == "status":
        show_status()
    elif args.command == "logs":
        show_logs(args.lines)


if __name__ == "__main__":
    main()
