#!/usr/bin/env python3
"""
Professional ComfyUI Error Diagnostic Tool for LoopyComfy
Identifies exact import errors, path issues, and dependency conflicts
"""

import sys
import os
import traceback
import importlib
import importlib.util
from pathlib import Path
import json

class ComfyUIDiagnostic:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
    def log_error(self, message, exception=None):
        error_info = {"message": message}
        if exception:
            error_info["exception"] = str(exception)
            error_info["traceback"] = traceback.format_exc()
        self.errors.append(error_info)
        
    def log_warning(self, message):
        self.warnings.append(message)
        
    def log_info(self, message):
        self.info.append(message)
        
    def check_python_environment(self):
        """Analyze Python environment and path configuration."""
        self.log_info("=== PYTHON ENVIRONMENT ANALYSIS ===")
        self.log_info(f"Python executable: {sys.executable}")
        self.log_info(f"Python version: {sys.version}")
        self.log_info(f"Platform: {sys.platform}")
        self.log_info(f"Current working directory: {os.getcwd()}")
        self.log_info(f"Script directory: {self.current_dir}")
        
        # Check if we're in ComfyUI embedded Python
        if "python_embeded" in sys.executable:
            self.log_info("OK Running in ComfyUI embedded Python environment")
        elif "ComfyUI" in os.getcwd():
            self.log_warning("Running in ComfyUI directory but not using embedded Python")
        else:
            self.log_warning("Not running in ComfyUI environment")
            
        # Analyze Python path
        self.log_info(f"Python path has {len(sys.path)} entries:")
        for i, path in enumerate(sys.path):
            self.log_info(f"  [{i}] {path}")
            
    def check_file_structure(self):
        """Verify LoopyComfy file structure is correct."""
        self.log_info("=== FILE STRUCTURE ANALYSIS ===")
        
        expected_files = [
            "nodes/video_asset_loader.py",
            "nodes/markov_sequencer.py", 
            "nodes/video_composer.py",
            "nodes/video_saver.py",
            "core/markov_engine.py",
            "utils/video_utils.py",
            "utils/security_utils.py",
            "utils/memory_manager.py",
            "utils/performance_optimizations.py",
            "requirements.txt",
            "__init__.py"
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = os.path.join(self.current_dir, file_path)
            if os.path.exists(full_path):
                self.log_info(f"OK Found: {file_path}")
            else:
                missing_files.append(file_path)
                self.log_error(f"MISSING: {file_path}")
                
        if missing_files:
            self.log_error(f"Missing {len(missing_files)} critical files")
        else:
            self.log_info("OK All expected files present")
            
    def check_dependencies(self):
        """Test all required dependencies individually."""
        self.log_info("=== DEPENDENCY ANALYSIS ===")
        
        dependencies = [
            ("numpy", "Mathematical operations"),
            ("cv2", "OpenCV for video processing"), 
            ("psutil", "System monitoring"),
            ("scipy", "Scientific computing"),
            ("PIL", "Pillow for image processing"),
            ("imageio", "Image/video I/O"),
            ("ffmpeg", "FFmpeg Python wrapper"),
            ("sklearn", "Scikit-learn for ML"),
            ("multiprocessing", "Built-in multiprocessing"),
            ("threading", "Built-in threading"),
            ("queue", "Built-in queue"),
            ("collections", "Built-in collections"),
            ("hashlib", "Built-in hashing"),
            ("hmac", "Built-in HMAC"),
            ("secrets", "Built-in cryptography"),
            ("tracemalloc", "Built-in memory tracing")
        ]
        
        failed_deps = []
        for dep_name, description in dependencies:
            try:
                module = importlib.import_module(dep_name)
                version = getattr(module, '__version__', 'unknown')
                self.log_info(f"OK {dep_name} ({version}): {description}")
            except ImportError as e:
                failed_deps.append(dep_name)
                self.log_error(f"ERROR {dep_name}: {description}", e)
                
        if failed_deps:
            self.log_error(f"Missing dependencies: {', '.join(failed_deps)}")
            return False
        else:
            self.log_info("OK All dependencies available")
            return True
            
    def test_utility_imports(self):
        """Test importing utility modules with detailed error reporting."""
        self.log_info("=== UTILITY MODULE IMPORT ANALYSIS ===")
        
        # Add current directory to path for testing
        if self.current_dir not in sys.path:
            sys.path.insert(0, self.current_dir)
            self.log_info(f"Added to sys.path: {self.current_dir}")
            
        utilities = [
            ("utils.video_utils", "Video processing utilities"),
            ("utils.security_utils", "Security validation utilities"),
            ("utils.memory_manager", "Memory management utilities"), 
            ("utils.performance_optimizations", "Performance optimization utilities"),
            ("utils.optimized_video_processing", "Optimized video processing"),
            ("core.markov_engine", "Markov chain engine")
        ]
        
        failed_utils = []
        for util_name, description in utilities:
            try:
                spec = importlib.util.find_spec(util_name)
                if spec is None:
                    self.log_error(f"ERROR {util_name}: Module spec not found")
                    failed_utils.append(util_name)
                    continue
                    
                module = importlib.import_module(util_name)
                self.log_info(f"OK {util_name}: {description}")
                
                # Test key classes/functions exist
                if util_name == "utils.security_utils":
                    if hasattr(module, 'PathValidator'):
                        self.log_info(f"    |-- PathValidator class: Available")
                    else:
                        self.log_warning(f"    |-- PathValidator class: Missing")
                        
                elif util_name == "core.markov_engine":
                    if hasattr(module, 'MarkovTransitionEngine'):
                        self.log_info(f"    |-- MarkovTransitionEngine class: Available")
                    else:
                        self.log_warning(f"    |-- MarkovTransitionEngine class: Missing")
                        
            except ImportError as e:
                failed_utils.append(util_name)
                self.log_error(f"ERROR {util_name}: {description}", e)
            except Exception as e:
                failed_utils.append(util_name)
                self.log_error(f"ERROR {util_name}: Unexpected error during import", e)
                
        return len(failed_utils) == 0
        
    def test_node_imports(self):
        """Test importing individual nodes with detailed error tracing."""
        self.log_info("=== NODE IMPORT ANALYSIS ===")
        
        nodes = [
            ("nodes.video_asset_loader", "LoopyComfy_VideoAssetLoader"),
            ("nodes.markov_sequencer", "LoopyComfy_MarkovVideoSequencer"),
            ("nodes.video_composer", "LoopyComfy_VideoSequenceComposer"), 
            ("nodes.video_saver", "LoopyComfy_VideoSaver")
        ]
        
        failed_nodes = []
        successful_nodes = {}
        
        for node_module, node_class in nodes:
            try:
                # Test module import
                module = importlib.import_module(node_module)
                self.log_info(f"OK {node_module}: Module imported successfully")
                
                # Test class exists
                if hasattr(module, node_class):
                    cls = getattr(module, node_class)
                    self.log_info(f"    |-- {node_class}: Class found")
                    
                    # Test ComfyUI interface methods
                    if hasattr(cls, 'INPUT_TYPES'):
                        try:
                            input_types = cls.INPUT_TYPES()
                            required_inputs = list(input_types.get('required', {}).keys())
                            self.log_info(f"    |-- INPUT_TYPES: {len(required_inputs)} inputs - {required_inputs[:3]}...")
                        except Exception as e:
                            self.log_error(f"    |-- INPUT_TYPES: Method call failed", e)
                    else:
                        self.log_error(f"    |-- INPUT_TYPES: Method not found")
                        
                    if hasattr(cls, 'RETURN_TYPES'):
                        return_types = getattr(cls, 'RETURN_TYPES', 'undefined')
                        self.log_info(f"    |-- RETURN_TYPES: {return_types}")
                    else:
                        self.log_warning(f"    |-- RETURN_TYPES: Attribute not found")
                        
                    successful_nodes[node_class] = cls
                else:
                    self.log_error(f"    |-- {node_class}: Class not found in module")
                    failed_nodes.append(node_module)
                    
            except ImportError as e:
                failed_nodes.append(node_module)
                self.log_error(f"ERROR {node_module}: Import failed", e)
            except Exception as e:
                failed_nodes.append(node_module)
                self.log_error(f"ERROR {node_module}: Unexpected error", e)
                
        return len(failed_nodes) == 0, successful_nodes
        
    def test_comfyui_registration(self, successful_nodes):
        """Test ComfyUI node registration process."""
        self.log_info("=== COMFYUI REGISTRATION ANALYSIS ===")
        
        try:
            # Test __init__.py import
            init_spec = importlib.util.find_spec("__init__")
            if init_spec is None:
                self.log_error("__init__.py module spec not found")
                return False
                
            # Import __init__ and check mappings
            import __init__ as init_module
            
            if hasattr(init_module, 'NODE_CLASS_MAPPINGS'):
                mappings = init_module.NODE_CLASS_MAPPINGS
                self.log_info(f"NODE_CLASS_MAPPINGS found with {len(mappings)} entries:")
                for name, cls in mappings.items():
                    self.log_info(f"  {name}: {cls}")
            else:
                self.log_error("NODE_CLASS_MAPPINGS not found in __init__.py")
                return False
                
            if hasattr(init_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                display_names = init_module.NODE_DISPLAY_NAME_MAPPINGS
                self.log_info(f"NODE_DISPLAY_NAME_MAPPINGS found with {len(display_names)} entries")
            else:
                self.log_warning("NODE_DISPLAY_NAME_MAPPINGS not found")
                
            return True
            
        except Exception as e:
            self.log_error("Failed to test ComfyUI registration", e)
            return False
            
    def run_full_diagnosis(self):
        """Run complete diagnostic analysis."""
        print("LOOPYCOMFY PROFESSIONAL DIAGNOSTIC ANALYSIS")
        print("=" * 80)
        
        # Run all diagnostic checks
        self.check_python_environment()
        self.check_file_structure()
        deps_ok = self.check_dependencies()
        utils_ok = self.test_utility_imports()
        nodes_ok, successful_nodes = self.test_node_imports()
        registration_ok = self.test_comfyui_registration(successful_nodes)
        
        # Generate comprehensive report
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        
        # Print all info messages
        for info in self.info:
            print(f"INFO: {info}")
            
        # Print warnings
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"WARN: {warning}")
                
        # Print errors with full detail
        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"ERROR: {error['message']}")
                if 'exception' in error:
                    print(f"  Exception: {error['exception']}")
                if 'traceback' in error:
                    print(f"  Traceback:\n{error['traceback']}")
                    
        # Final assessment
        print("\n" + "=" * 80)
        print("FINAL ASSESSMENT")
        print("=" * 80)
        
        if len(self.errors) == 0:
            print("SUCCESS NO CRITICAL ERRORS FOUND")
            print("All systems appear to be working correctly.")
            print("If nodes still don't load, the issue may be in ComfyUI's node discovery process.")
        else:
            print(f"FAIL {len(self.errors)} CRITICAL ERRORS FOUND")
            print("These errors must be resolved for nodes to load properly.")
            
            # Provide specific recommendations based on error types
            error_messages = [e['message'] for e in self.errors]
            
            if any('Missing' in msg and '.py' in msg for msg in error_messages):
                print("\nRECOMMENDATION: File structure issues detected")
                print("- Ensure all Python files are present in correct directories")
                print("- Verify git pull completed successfully")
                
            if any('ImportError' in str(e.get('exception', '')) for e in self.errors):
                print("\nRECOMMENDATION: Import/dependency issues detected")  
                print("- Install missing dependencies in ComfyUI Python environment")
                print("- Check for conflicting module versions")
                
            if any('PathValidator' in msg or 'MarkovTransitionEngine' in msg for msg in error_messages):
                print("\nRECOMMENDATION: Core class definition issues")
                print("- Verify utility modules are complete and not corrupted")
                print("- Check for syntax errors in Python files")
                
        return len(self.errors) == 0

if __name__ == "__main__":
    diagnostic = ComfyUIDiagnostic()
    success = diagnostic.run_full_diagnosis()
    
    # Generate JSON report for programmatic analysis
    report = {
        "success": success,
        "errors": diagnostic.errors,
        "warnings": diagnostic.warnings, 
        "info": diagnostic.info,
        "python_executable": sys.executable,
        "current_directory": os.getcwd()
    }
    
    with open("diagnostic_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDetailed report saved to: diagnostic_report.json")
    sys.exit(0 if success else 1)