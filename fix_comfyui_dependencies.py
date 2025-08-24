#!/usr/bin/env python3
"""
Simple ComfyUI Dependency Fixer for LoopyComfy
Run this directly in your ComfyUI environment
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {cmd}")
        print(f"Error: {e}")
        return False

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("LOOPYCOMFY DEPENDENCY FIXER")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print()

    # Required dependencies
    dependencies = [
        ("numpy", "numpy>=1.24.0,<2.0.0"),
        ("cv2", "opencv-python>=4.8.0,<5.0.0"), 
        ("psutil", "psutil>=5.9.6,<6.0.0"),
        ("scipy", "scipy>=1.10.0,<2.0.0"),
        ("PIL", "Pillow>=10.0.0,<11.0.0"),
        ("imageio", "imageio>=2.30.0,<3.0.0"),
        ("ffmpeg", "ffmpeg-python>=0.2.0,<1.0.0"),
        ("sklearn", "scikit-learn>=1.3.0,<2.0.0")
    ]

    # Test current status
    print("Current dependency status:")
    missing = []
    for module_name, pip_name in dependencies:
        if test_import(module_name):
            print(f"  ✓ {module_name} - OK")
        else:
            print(f"  ✗ {module_name} - MISSING")
            missing.append((module_name, pip_name))

    if not missing:
        print("\n✓ All dependencies already installed!")
        return

    print(f"\nInstalling {len(missing)} missing dependencies...")
    
    # Upgrade pip first
    print("Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Install missing dependencies
    success_count = 0
    for module_name, pip_name in missing:
        print(f"\nInstalling {pip_name}...")
        if run_command(f"{sys.executable} -m pip install \"{pip_name}\""):
            # Test if it worked
            if test_import(module_name):
                print(f"  ✓ {module_name} - Successfully installed")
                success_count += 1
            else:
                print(f"  ⚠ {module_name} - Installed but import still fails")
        else:
            print(f"  ✗ {module_name} - Installation failed")

    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    print(f"Successfully installed: {success_count}/{len(missing)} dependencies")
    
    if success_count == len(missing):
        print("\n✅ SUCCESS: All dependencies installed!")
        print("\nNext steps:")
        print("1. Restart ComfyUI completely")
        print("2. LoopyComfy nodes should now load correctly")
        print("3. Look for nodes: Video Asset Loader, Markov Video Sequencer, etc.")
    else:
        print(f"\n⚠ {len(missing) - success_count} dependencies still missing")
        print("\nTroubleshooting:")
        print("1. Run as Administrator (Windows)")
        print("2. Check internet connection") 
        print("3. Update pip: python -m pip install --upgrade pip")
        print("4. Try installing manually: pip install psutil scipy scikit-learn")

if __name__ == "__main__":
    main()