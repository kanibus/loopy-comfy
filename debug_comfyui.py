#!/usr/bin/env python3
"""
ComfyUI Custom Node Debug Script
Helps diagnose why LoopyComfy nodes are not loading in ComfyUI
"""

import sys
import os
from pathlib import Path

def check_comfyui_installation():
    """Check if ComfyUI is properly installed and configured"""
    print("=== COMFYUI CUSTOM NODE DIAGNOSTIC ===\n")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in a ComfyUI custom nodes directory
    possible_comfyui_paths = [
        current_dir.parent / "ComfyUI",
        current_dir.parent.parent / "ComfyUI", 
        Path.home() / "ComfyUI",
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
    ]
    
    comfyui_found = False
    for path in possible_comfyui_paths:
        if path.exists() and (path / "main.py").exists():
            print(f"[SUCCESS] ComfyUI found at: {path}")
            comfyui_found = True
            
            # Check custom_nodes directory
            custom_nodes_dir = path / "custom_nodes"
            if custom_nodes_dir.exists():
                print(f"[SUCCESS] custom_nodes directory: {custom_nodes_dir}")
                
                # List existing custom nodes
                custom_nodes = [d.name for d in custom_nodes_dir.iterdir() if d.is_dir()]
                print(f"  Existing custom nodes: {custom_nodes}")
                
                # Check if loopy-comfy is installed
                loopy_dir = custom_nodes_dir / "loopy-comfy"
                if loopy_dir.exists():
                    print(f"[SUCCESS] loopy-comfy found in custom_nodes: {loopy_dir}")
                    
                    # Check __init__.py
                    init_file = loopy_dir / "__init__.py"
                    if init_file.exists():
                        print(f"[SUCCESS] __init__.py exists: {init_file}")
                    else:
                        print(f"[ERROR] __init__.py missing: {init_file}")
                else:
                    print(f"[ERROR] loopy-comfy NOT found in custom_nodes")
                    print(f"  Expected location: {loopy_dir}")
                    print(f"  SOLUTION: Copy/symlink this directory to ComfyUI/custom_nodes/")
            break
    
    if not comfyui_found:
        print("[ERROR] ComfyUI installation not found")
        print("  Common locations checked:")
        for path in possible_comfyui_paths:
            print(f"    - {path}")
    
    print("\n=== INSTALLATION INSTRUCTIONS ===")
    if comfyui_found:
        for path in possible_comfyui_paths:
            if path.exists() and (path / "main.py").exists():
                target_dir = path / "custom_nodes" / "loopy-comfy"
                print(f"1. Copy this entire directory to: {target_dir}")
                print(f"2. Or create symlink:")
                print(f"   Windows: mklink /D \"{target_dir}\" \"{current_dir}\"")
                print(f"   Linux/Mac: ln -s \"{current_dir}\" \"{target_dir}\"")
                break
    else:
        print("1. First install ComfyUI: https://github.com/comfyanonymous/ComfyUI")
        print("2. Then copy this directory to ComfyUI/custom_nodes/loopy-comfy")
    
    print(f"3. Restart ComfyUI")
    print(f"4. Look for nodes under 'LoopyComfy' category")

def test_node_creation():
    """Test if nodes can be instantiated"""
    print("\n=== NODE INSTANTIATION TEST ===")
    
    # Add current directory to path
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
        from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer  
        from nodes.video_composer import LoopyComfy_VideoSequenceComposer
        from nodes.video_saver import LoopyComfy_VideoSaver
        
        nodes = [
            ("VideoAssetLoader", LoopyComfy_VideoAssetLoader),
            ("MarkovVideoSequencer", LoopyComfy_MarkovVideoSequencer),
            ("VideoSequenceComposer", LoopyComfy_VideoSequenceComposer),
            ("VideoSaver", LoopyComfy_VideoSaver),
        ]
        
        for name, node_class in nodes:
            try:
                # Test INPUT_TYPES method
                input_types = node_class.INPUT_TYPES()
                print(f"[SUCCESS] {name}: INPUT_TYPES = {list(input_types.keys())}")
                
                # Test RETURN_TYPES if available
                if hasattr(node_class, 'RETURN_TYPES'):
                    return_types = node_class.RETURN_TYPES
                    print(f"  RETURN_TYPES = {return_types}")
                    
                # Test CATEGORY if available  
                if hasattr(node_class, 'CATEGORY'):
                    category = node_class.CATEGORY
                    print(f"  CATEGORY = {category}")
                    
            except Exception as e:
                print(f"[ERROR] {name}: Error - {e}")
                
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")

if __name__ == "__main__":
    check_comfyui_installation()
    test_node_creation()