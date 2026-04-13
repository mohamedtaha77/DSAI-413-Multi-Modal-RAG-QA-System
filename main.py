# main.py
# Entry point: ingest a document or launch the QA interface.

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="RAG-Based QA System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into the vector store")
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")

    subparsers.add_parser("serve", help="Launch the Streamlit QA interface")

    subparsers.add_parser("benchmark", help="Run the evaluation benchmark")

    args = parser.parse_args()

    if args.command == "ingest":
        from ingestion.embedder import ingest_pdf
        ingest_pdf(args.pdf_path)

    elif args.command == "serve":
        app_path = str(__file__).replace("main.py", "interface/app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)

    elif args.command == "benchmark":
        from evaluation.benchmark import run_benchmark
        run_benchmark()


if __name__ == "__main__":
    main()
