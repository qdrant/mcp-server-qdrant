#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    """Install the package in development mode."""
    print("Installing mcp-server-qdrant with updated dependencies...")
    
    # Get the current directory
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Run the installation command
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Installation successful!")
        print(result.stdout)
    else:
        print(f"Installation failed with code {result.returncode}")
        print(f"Error: {result.stderr}")
        
if __name__ == "__main__":
    main()
