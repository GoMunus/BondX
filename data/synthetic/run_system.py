#!/usr/bin/env python3
"""
BondX Synthetic Dataset System Launcher
Simple interface to run the synthetic data pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main launcher function."""
    print("BondX Synthetic Dataset System")
    print("=" * 50)
    
    # Change to the synthetic data directory
    os.chdir(Path(__file__).parent)
    
    print("1. Generating synthetic dataset...")
    try:
        result = subprocess.run([sys.executable, 'generate_synthetic_dataset.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        print("✓ Dataset generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset generation failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return
    
    print("\n2. Running validation tests...")
    try:
        result = subprocess.run([sys.executable, 'test_dataset.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        print("✓ All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return
    
    print("\n" + "="*50)
    print("🎉 SYNTHETIC DATA SYSTEM COMPLETE!")
    print("="*50)
    print("Generated files:")
    print("- bondx_issuers_260.csv")
    print("- bondx_issuers_260.jsonl")
    print("- README.md")
    print("\nFiles are ready for use in BondX development and testing!")

if __name__ == "__main__":
    main()
