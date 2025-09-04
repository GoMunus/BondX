#!/usr/bin/env python3
"""
Python script to execute git commands for committing all changes
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and display the result"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.stdout:
            print("✅ Output:")
            print(result.stdout)
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function to execute git commands"""
    print("🚀 Starting git operations for BondX...")
    
    # Change to BondX directory
    os.chdir(r"C:\Users\doall\BondX")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Step 1: Add all changes
    if not run_command("git add -A", "Adding all changes to git"):
        print("❌ Failed to add changes")
        return
    
    # Step 2: Check status
    if not run_command("git status --short", "Checking git status"):
        print("❌ Failed to check status")
        return
    
    # Step 3: Commit changes
    commit_message = "Complete BondX implementation: MLOps, AI components, trading engine, and comprehensive platform"
    if not run_command(f'git commit -m "{commit_message}"', "Committing all changes"):
        print("❌ Failed to commit changes")
        return
    
    # Step 4: Push to GitHub
    if not run_command("git push origin main", "Pushing to GitHub"):
        print("❌ Failed to push to GitHub")
        return
    
    print("\n🎉 All git operations completed successfully!")
    print("✅ Changes committed and pushed to GitHub")

if __name__ == "__main__":
    main()

