#!/usr/bin/env python3
"""
API Extensions for LoopyComfy Non-Linear Video Avatar

This module provides API endpoints for UI integration, including folder browsing
functionality and enhanced ComfyUI integration.

Note: This is a placeholder implementation. For full functionality, integrate
with ComfyUI's server architecture or create a standalone Flask/FastAPI server.
"""

import os
import json
from typing import Dict, Any, Optional


class LoopyComfyAPI:
    """API endpoints for LoopyComfy UI integration."""
    
    def __init__(self):
        self.allowed_base_paths = [
            os.getcwd(),
            os.path.expanduser("~"),
            "/tmp" if os.name != 'nt' else os.environ.get('TEMP', 'C:\\temp')
        ]
    
    def browse_folder(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle folder browsing requests from UI.
        
        Args:
            request_data: Request data with node_id and current_path
            
        Returns:
            Response data with selected path or error
        """
        try:
            current_path = request_data.get('current_path', os.getcwd())
            node_id = request_data.get('node_id', 'unknown')
            
            # Security validation
            if not self._is_path_allowed(current_path):
                return {
                    "error": "Path not allowed for security reasons",
                    "path": None
                }
            
            # For demonstration, return a mock folder selection
            # In real implementation, this would open native OS dialog
            
            # Try to use tkinter for folder selection
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                # Create hidden root window
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                selected_path = filedialog.askdirectory(
                    title="Select Video Folder",
                    initialdir=current_path if os.path.exists(current_path) else os.getcwd()
                )
                
                root.destroy()
                
                if selected_path:
                    return {
                        "success": True,
                        "path": selected_path,
                        "message": f"Selected folder: {os.path.basename(selected_path)}"
                    }
                else:
                    return {
                        "success": False,
                        "path": None,
                        "message": "No folder selected"
                    }
                    
            except ImportError:
                # Fallback when tkinter not available
                return {
                    "success": False,
                    "error": "Native folder dialog not available",
                    "path": None,
                    "fallback_suggestions": [
                        "./assets/videos/",
                        "./input/",
                        "./ComfyUI/input/",
                        os.path.expanduser("~/Videos/") if os.path.exists(os.path.expanduser("~/Videos/")) else "./videos/",
                        os.path.expanduser("~/Downloads/") if os.path.exists(os.path.expanduser("~/Downloads/")) else "./downloads/"
                    ]
                }
                
        except Exception as e:
            return {
                "error": f"Failed to open folder dialog: {str(e)}",
                "path": None
            }
    
    def _is_path_allowed(self, path: str) -> bool:
        """
        Check if path is allowed for security.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is allowed
        """
        try:
            abs_path = os.path.abspath(path)
            
            # Check against allowed base paths
            for base_path in self.allowed_base_paths:
                if abs_path.startswith(os.path.abspath(base_path)):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def get_video_info(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get video information for preview.
        
        Args:
            request_data: Request with video path
            
        Returns:
            Video metadata or error
        """
        try:
            video_path = request_data.get('video_path')
            
            if not video_path or not os.path.exists(video_path):
                return {"error": "Video file not found"}
            
            # Security check
            if not self._is_path_allowed(video_path):
                return {"error": "Video path not allowed"}
            
            # Extract basic info using OpenCV
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                return {
                    "success": True,
                    "info": {
                        "filename": os.path.basename(video_path),
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "duration": duration,
                        "frame_count": frame_count,
                        "resolution": f"{width}x{height}",
                        "aspect_ratio": round(width / height, 2) if height > 0 else 1.0
                    }
                }
            finally:
                cap.release()
                
        except Exception as e:
            return {"error": f"Failed to get video info: {str(e)}"}
    
    def browse_output_directory(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle output directory browsing requests from UI.
        
        Args:
            request_data: Request data with current output directory
            
        Returns:
            Response data with selected directory or error
        """
        try:
            current_path = request_data.get('current_path', './output/')
            
            # Security validation
            if not self._is_path_allowed(current_path):
                return {
                    "error": "Path not allowed for security reasons",
                    "path": None
                }
            
            # Try to use tkinter for directory selection
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                # Create hidden root window
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                selected_path = filedialog.askdirectory(
                    title="Select Output Directory - LoopyComfy",
                    initialdir=current_path if os.path.exists(current_path) else os.getcwd(),
                    mustexist=False  # Allow creating new directories
                )
                
                root.destroy()
                
                if selected_path:
                    # Ensure directory exists and is writable
                    try:
                        os.makedirs(selected_path, exist_ok=True)
                        is_writable = os.access(selected_path, os.W_OK)
                        
                        return {
                            "success": True,
                            "path": selected_path,
                            "directory_path": selected_path,  # For compatibility
                            "folder_name": os.path.basename(selected_path),
                            "is_writable": is_writable,
                            "message": f"Selected output directory: {os.path.basename(selected_path)}"
                        }
                    except Exception as create_error:
                        return {
                            "success": True,
                            "path": selected_path,
                            "directory_path": selected_path,
                            "folder_name": os.path.basename(selected_path),
                            "is_writable": False,
                            "warning": f"Could not create/access directory: {str(create_error)}",
                            "message": f"Selected directory (may need manual creation): {os.path.basename(selected_path)}"
                        }
                else:
                    return {
                        "success": False,
                        "path": None,
                        "message": "No directory selected"
                    }
                    
            except ImportError:
                # Fallback when tkinter not available
                return {
                    "success": False,
                    "error": "Native directory dialog not available",
                    "path": None,
                    "fallback_suggestions": [
                        "./output/",
                        "./renders/",
                        "./ComfyUI/output/",
                        os.path.expanduser("~/Documents/LoopyComfy/") if os.path.exists(os.path.expanduser("~/Documents/")) else "./generated/",
                        os.path.expanduser("~/Desktop/LoopyComfy/") if os.path.exists(os.path.expanduser("~/Desktop/")) else "./desktop/"
                    ]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to open directory dialog: {str(e)}",
                "path": None
            }


# Mock server integration functions
def register_api_routes(server_instance):
    """
    Register API routes with ComfyUI server or standalone server.
    
    This is a placeholder - actual implementation depends on the server framework.
    """
    api = LoopyComfyAPI()
    
    # Example for Flask integration:
    # @server_instance.route('/loopycomfy/browse_folder', methods=['POST'])
    # def handle_browse_folder():
    #     return api.browse_folder(request.get_json())
    
    print("LoopyComfy API routes registered (placeholder)")


def create_standalone_server(host: str = "localhost", port: int = 8188):
    """
    Create standalone API server for LoopyComfy.
    
    This can be used if ComfyUI server integration is not available.
    """
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)  # Enable CORS for browser requests
        
        api = LoopyComfyAPI()
        
        @app.route('/loopycomfy/browse_folder', methods=['POST'])
        def browse_folder():
            return jsonify(api.browse_folder(request.get_json() or {}))
        
        @app.route('/loopycomfy/browse_output_dir', methods=['POST'])
        def browse_output_directory():
            return jsonify(api.browse_output_directory(request.get_json() or {}))
        
        @app.route('/loopycomfy/video_info', methods=['POST'])
        def video_info():
            return jsonify(api.get_video_info(request.get_json() or {}))
        
        @app.route('/loopycomfy/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy", 
                "service": "LoopyComfy API",
                "version": "1.2.0",
                "features": ["folder_browser", "output_directory_browser", "video_info", "format_validation"]
            })
        
        print(f"LoopyComfy API server starting on {host}:{port}")
        app.run(host=host, port=port, debug=False)
        
    except ImportError:
        print("Flask not available - cannot create standalone server")
        print("Install Flask with: pip install flask flask-cors")
        return None


if __name__ == "__main__":
    # Run standalone server for testing
    create_standalone_server()