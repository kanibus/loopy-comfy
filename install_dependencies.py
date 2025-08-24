#!/usr/bin/env python3
"""
LoopyComfy Dependency Installer for ComfyUI
Automatically detects and installs missing dependencies
"""

import subprocess
import sys
import importlib

def check_dependency(module_name, pip_name=None):
    """Check if a dependency is installed."""
    if pip_name is None:
        pip_name = module_name
    
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")
        return False

def main():
    """Main installation routine."""
    print("=" * 60)
    print("LOOPYCOMFY DEPENDENCY INSTALLER")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("")
    
    # List of required dependencies
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
    
    missing_deps = []
    installed_deps = []
    
    # Check each dependency
    print("Checking dependencies:")
    for module_name, pip_package in dependencies:
        is_installed, error = check_dependency(module_name)
        if is_installed:
            print(f"  ✓ {module_name} - Already installed")
            installed_deps.append(module_name)
        else:
            print(f"  ✗ {module_name} - Missing")
            missing_deps.append((module_name, pip_package))
    
    print("")
    print(f"Dependencies status: {len(installed_deps)} installed, {len(missing_deps)} missing")
    
    # Install missing dependencies
    if missing_deps:
        print("\\nInstalling missing dependencies...")
        success_count = 0
        
        for module_name, pip_package in missing_deps:
            if install_package(pip_package):
                # Verify installation
                is_installed, _ = check_dependency(module_name)
                if is_installed:
                    print(f"  ✓ {module_name} - Successfully installed")
                    success_count += 1
                else:
                    print(f"  ✗ {module_name} - Installation failed")
            else:
                print(f"  ✗ {module_name} - Installation failed")
        
        print("")
        print(f"Installation complete: {success_count}/{len(missing_deps)} successful")
        
        if success_count == len(missing_deps):
            print("✓ All dependencies installed successfully!")
            print("\\nNext steps:")
            print("1. Restart ComfyUI")
            print("2. The LoopyComfy nodes should now load correctly")
        else:
            print("✗ Some dependencies failed to install")
            print("\\nTroubleshooting:")
            print("1. Run this script as Administrator (Windows)")
            print("2. Check network connectivity")
            print("3. Update pip: python -m pip install --upgrade pip")
            print("4. Install manually: pip install -r requirements.txt")
    else:
        print("✓ All dependencies are already installed!")
        print("\\nIf nodes still fail to load:")
        print("1. Restart ComfyUI completely")
        print("2. Check ComfyUI console for specific error messages")
        print("3. Verify LoopyComfy is in ComfyUI/custom_nodes/ directory")
    
    print("=" * 60)

if __name__ == "__main__":
    main()