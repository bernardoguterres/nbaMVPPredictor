#!/usr/bin/env python3
"""
NBA MVP Predictor - Main Pipeline
Runs the complete end-to-end pipeline for NBA MVP prediction

Usage:
    python run_pipeline.py [--skip-scraping] [--skip-processing] [--skip-ml]

Options:
    --skip-scraping     Skip data scraping step (use existing HTML/CSV files)
    --skip-processing   Skip data processing step (use existing player_mvp_stats.csv)
    --skip-ml           Skip machine learning step
    --scraping-only     Only run scraping, skip processing and ML
    --processing-only   Only run processing, skip scraping and ML
    --ml-only           Only run ML, skip scraping and processing
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from datetime import datetime


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='NBA MVP Prediction Pipeline')
    parser.add_argument('--skip-scraping', action='store_true',
                        help='Skip data scraping step')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip data processing step')
    parser.add_argument('--skip-ml', action='store_true',
                        help='Skip machine learning step')
    parser.add_argument('--scraping-only', action='store_true',
                        help='Only run scraping')
    parser.add_argument('--processing-only', action='store_true',
                        help='Only run processing')
    parser.add_argument('--ml-only', action='store_true',
                        help='Only run machine learning')
    return parser.parse_args()


def run_scraping():
    """Run data scraping"""
    print("\n" + "=" * 70)
    print("STEP 1: DATA SCRAPING")
    print("=" * 70)
    from datascraping import main as scraping_main
    scraping_main()


def run_processing():
    """Run data processing"""
    print("\n" + "=" * 70)
    print("STEP 2: DATA PROCESSING")
    print("=" * 70)
    from predictors import main as predictors_main
    predictors_main()


def run_ml():
    """Run machine learning"""
    print("\n" + "=" * 70)
    print("STEP 3: MACHINE LEARNING")
    print("=" * 70)
    from machine_learning import main as ml_main
    ml_main()


def main():
    """Main pipeline execution"""
    args = parse_args()

    start_time = datetime.now()

    print("=" * 70)
    print("NBA MVP PREDICTOR - FULL PIPELINE")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Determine which steps to run
    run_steps = {
        'scraping': not args.skip_scraping,
        'processing': not args.skip_processing,
        'ml': not args.skip_ml
    }

    # Handle "only" flags
    if args.scraping_only:
        run_steps = {'scraping': True, 'processing': False, 'ml': False}
    elif args.processing_only:
        run_steps = {'scraping': False, 'processing': True, 'ml': False}
    elif args.ml_only:
        run_steps = {'scraping': False, 'processing': False, 'ml': True}

    try:
        # Step 1: Data Scraping
        if run_steps['scraping']:
            run_scraping()
        else:
            print("\n⏭️  Skipping data scraping (using existing data)")

        # Step 2: Data Processing
        if run_steps['processing']:
            run_processing()
        else:
            print("\n⏭️  Skipping data processing (using existing data)")

        # Step 3: Machine Learning
        if run_steps['ml']:
            run_ml()
        else:
            print("\n⏭️  Skipping machine learning")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print("\n✅ All steps completed successfully!")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
