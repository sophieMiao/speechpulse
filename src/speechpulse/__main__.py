"""CLI entry point for SpeechPulse MCP Server.

This module provides the command-line interface for starting the SpeechPulse
MCP server. It can be invoked using:

    python -m speechpulse

Or directly:

    speechpulse-server
"""

import argparse
import logging
import sys

from .server import mcp


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the server.
    
    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="speechpulse",
        description="SpeechPulse MCP Server - Voice emotion understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start server with stdio transport
  %(prog)s --transport sse          # Start server with SSE transport
  %(prog)s --port 8080              # Use custom port for SSE transport
  %(prog)s -v                       # Enable verbose logging

For more information, visit: https://github.com/yourusername/speechpulse
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s 0.1.0",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for SSE transport (default: 127.0.0.1)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for SSE transport (default: 8000)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SpeechPulse MCP Server...")
    logger.info(f"Transport: {args.transport}")
    
    try:
        if args.transport == "stdio":
            # Run with stdio transport (default for MCP)
            mcp.run(transport="stdio")
        elif args.transport == "sse":
            # Run with SSE transport
            logger.info(f"Binding to {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
