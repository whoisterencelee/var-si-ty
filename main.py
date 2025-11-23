#!/usr/bin/env python3
"""
Main script to run the entire visualization final project pipeline.
This orchestrates all the individual scripts to perform the complete data processing workflow.
"""

import os
import sys
import subprocess

def run_script(script_name):
    """Run a Python script and check for errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}")

    try:
        # Run the script
        result = subprocess.run([sys.executable, f"scripts/{script_name}"], check=True)
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ {script_name} not found")
        return False

def main():
    """Run the complete pipeline."""
    print("VISUALIZATION FINAL PROJECT - COMPLETE PIPELINE")
    print("="*80)
    print("This will run all scripts in sequence to process events and courses data.")
    print("Make sure you have the required dependencies installed.")
    print("Run: pip install -r requirements.txt")
    print("="*80)

    # Define the scripts to run in order
    scripts = [
        "data_fetch_events.py",      # Fetch events data
        "data_clean_events.py",      # Clean and process events
        "visualize_events.py",       # Create event visualizations
        "extract_organized_by.py",   # Extract organizedBy values
        "faculty_detection.py",      # Detect faculties
        "visualize_faculty.py",      # Create faculty visualizations
        "data_fetch_courses.py",     # Fetch courses data
        "data_clean_courses.py",     # Clean courses data
        "topic_modeling.py"          # Run BERTopic modeling
    ]

    # Check if data and scripts directories exist
    if not os.path.exists("data"):
        print("❌ 'data' directory not found. Please create it first.")
        return

    if not os.path.exists("scripts"):
        print("❌ 'scripts' directory not found. Please create it first.")
        return

    # Run each script
    failed_scripts = []
    for script in scripts:
        if not run_script(script):
            failed_scripts.append(script)
            # Continue with other scripts even if one fails

    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")

    if not failed_scripts:
        print("✅ All scripts completed successfully!")
        print("\nGenerated files:")
        print("  • data/events_data_all.json")
        print("  • data/events_data_all.csv")
        print("  • data/events_en_US.csv, events_zh_TW.csv, events_pt_PT.csv")
        print("  • visuals/events_per_day_by_year_*.png (visualizations)")
        print("  • data/organizedBy_by_language_comparison.csv")
        print("  • data/events_en_US_cleaned.csv")
        print("  • visuals/faculty_distribution.png, visuals/monthly_distribution.png, visuals/faculty_month_heatmap.png")
        print("  • data/courses_catalog_data_all.json")
        print("  • data/courses_data_all.csv")
        print("  • data/courses_with_topics.csv")
        print("  • data/bertopic_model/ (model directory)")
    else:
        print(f"❌ {len(failed_scripts)} script(s) failed:")
        for script in failed_scripts:
            print(f"   • {script}")
        print("\nPlease check the error messages above and fix any issues.")

if __name__ == "__main__":
    main()