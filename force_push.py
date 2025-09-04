import subprocess
import os

print("🚀 Force pushing enhanced MLOps components to GitHub...")

try:
    # Change to BondX directory
    os.chdir(r"C:\Users\doall\BondX")
    
    # Execute force push
    result = subprocess.run(
        ["git", "push", "--force", "origin", "main"], 
        capture_output=True, 
        text=True, 
        cwd=os.getcwd()
    )
    
    if result.returncode == 0:
        print("✅ Successfully pushed to GitHub!")
        print("Output:", result.stdout)
    else:
        print("❌ Push failed!")
        print("Error:", result.stderr)
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\nDone! Check your GitHub repository.")
