"""
CLI Entrypoint for running the ShieldML complete training pipeline.
Usage:
    python run_pipeline.py
"""
import sys
import argparse
from src.pipeline import run_end_to_end_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run ShieldML Full Pipeline")
    parser.add_argument("--force-generate", action="store_true", help="Force regenerate raw transaction dataset")
    args = parser.parse_args()
    
    run_end_to_end_pipeline(force_generate=args.force_generate)

if __name__ == "__main__":
    main()
