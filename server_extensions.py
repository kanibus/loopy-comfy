# -*- coding: utf-8 -*-
"""
ComfyUI Server Extensions for LoopyComfy

This module provides server-side API routes that integrate with ComfyUI's server
for enhanced UI functionality including native folder browsing dialogs.
"""

import os
import json
from aiohttp import web
from typing import Dict, Any, Optional

# Import tkinter for cross-platform folder dialogs
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class LoopyComfyServerExtensions:
    """
    Server extensions for LoopyComfy that integrate with ComfyUI's web server.
    """
    
    def __init__(self):
        """Initialize server extensions."""
        self.routes = []
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes for LoopyComfy."""
        self.routes = [
            web.post('/loopycomfy/browse_folder', self.handle_browse_folder),
            web.post('/loopycomfy/browse_output_dir', self.handle_browse_output_dir),
            web.get('/loopycomfy/health', self.handle_health_check),
            web.get('/loopycomfy/system_info', self.handle_system_info)
        ]
    
    async def handle_browse_folder(self, request):
        """
        Handle folder browsing request for video input directory.
        """
        try:
            data = await request.json()
            current_path = data.get('current_path', os.getcwd())
            node_id = data.get('node_id', 'unknown')
            
            print(f"LoopyComfy: Handling folder browse request for node {node_id}")
            
            if not TKINTER_AVAILABLE:
                return web.json_response({
                    "success": False,
                    "error": "Native folder dialog not available",
                    "fallback_suggestions": self._get_common_video_paths(),
                    "message": "Folder browser requires tkinter. Please install tkinter or use manual path entry."
                })
            
            # Open folder dialog in a separate thread to avoid blocking
            result = await self._run_folder_dialog(
                title="Select Video Folder - LoopyComfy",
                initial_dir=current_path,
                must_exist=True
            )
            
            if result.get('success') and result.get('path'):
                # Count video files
                video_count = self._count_video_files(result['path'])
                result['video_count'] = video_count
                result['message'] = f"Selected folder with {video_count} video files"
                
                print(f"LoopyComfy: Selected folder '{result['path']}' with {video_count} videos")
            
            return web.json_response(result)
            
        except Exception as e:
            print(f"LoopyComfy API Error: {str(e)}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "fallback_suggestions": self._get_common_video_paths()
            }, status=500)
    
    async def handle_browse_output_dir(self, request):
        """
        Handle output directory browsing request.
        """
        try:
            data = await request.json()
            current_path = data.get('current_path', './output/')
            
            print(f"LoopyComfy: Handling output directory browse request")
            
            if not TKINTER_AVAILABLE:
                return web.json_response({
                    "success": False,
                    "error": "Native directory dialog not available",
                    "fallback_suggestions": self._get_common_output_paths(),
                    "message": "Directory browser requires tkinter. Please install tkinter or use manual path entry."
                })
            
            # Open directory dialog
            result = await self._run_folder_dialog(
                title="Select Output Directory - LoopyComfy",
                initial_dir=current_path,
                must_exist=False
            )
            
            if result.get('success') and result.get('path'):
                # Ensure directory exists and check writability
                try:
                    os.makedirs(result['path'], exist_ok=True)
                    is_writable = os.access(result['path'], os.W_OK)
                    
                    result['is_writable'] = is_writable
                    result['directory_path'] = result['path']  # For compatibility
                    result['folder_name'] = os.path.basename(result['path'])
                    result['message'] = f"Selected output directory: {result['path']}"
                    
                    if not is_writable:
                        result['warning'] = "Directory may not be writable"
                    
                    print(f"LoopyComfy: Selected output directory '{result['path']}'")
                    
                except Exception as create_error:
                    result['warning'] = f"Could not create directory: {str(create_error)}"
                    result['is_writable'] = False
            
            return web.json_response(result)
            
        except Exception as e:
            print(f"LoopyComfy API Error: {str(e)}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "fallback_suggestions": self._get_common_output_paths()
            }, status=500)
    
    async def handle_health_check(self, request):
        """Handle health check request."""
        return web.json_response({
            "status": "healthy",
            "service": "LoopyComfy Server Extensions",
            "version": "1.2.0",
            "features": [
                "folder_browser",
                "output_directory_browser", 
                "native_dialogs",
                "cross_platform"
            ],
            "tkinter_available": TKINTER_AVAILABLE
        })
    
    async def handle_system_info(self, request):
        """Handle system information request."""
        try:
            import platform
            
            return web.json_response({
                "success": True,
                "system": {
                    "platform": platform.system(),
                    "version": platform.version(),
                    "architecture": platform.architecture()[0],
                    "python_version": platform.python_version(),
                    "tkinter_available": TKINTER_AVAILABLE
                },
                "capabilities": {
                    "folder_browser": TKINTER_AVAILABLE,
                    "cross_platform_paths": True,
                    "native_dialogs": TKINTER_AVAILABLE
                },
                "paths": {
                    "current_dir": os.getcwd(),
                    "common_video_paths": self._get_common_video_paths(),
                    "common_output_paths": self._get_common_output_paths()
                }
            })
            
        except Exception as e:
            print(f"LoopyComfy API Error: {str(e)}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _run_folder_dialog(self, title: str, initial_dir: str, must_exist: bool = True):
        """
        Run folder selection dialog in a thread-safe manner.
        """
        if not TKINTER_AVAILABLE:
            return {"success": False, "error": "Tkinter not available"}
        
        try:
            # Create root window
            root = tk.Tk()
            root.withdraw()  # Hide main window
            root.attributes('-topmost', True)  # Bring to front
            root.lift()
            
            # Ensure initial directory exists
            if not os.path.exists(initial_dir):
                initial_dir = os.getcwd()
            
            # Open dialog
            folder_path = filedialog.askdirectory(
                title=title,
                initialdir=initial_dir,
                mustexist=must_exist
            )
            
            # Cleanup
            root.destroy()
            
            if folder_path:
                return {
                    "success": True,
                    "path": os.path.abspath(folder_path),
                    "folder_name": os.path.basename(folder_path),
                    "exists": os.path.exists(folder_path)
                }
            else:
                return {"success": False, "cancelled": True}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _count_video_files(self, directory: str) -> int:
        """Count video files in directory."""
        try:
            video_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv', '.wmv'}
            
            video_count = 0
            for file in os.listdir(directory):
                if os.path.splitext(file)[1].lower() in video_extensions:
                    video_count += 1
            
            return video_count
            
        except Exception as e:
            print(f"Warning: Could not count video files in {directory}: {str(e)}")
            return 0
    
    def _get_common_video_paths(self):
        """Get list of common video directory paths."""
        common_paths = [
            "./assets/videos/",
            "./input/",
            "./ComfyUI/input/",
            "./videos/"
        ]
        
        # Add platform-specific paths
        if os.name == 'nt':  # Windows
            common_paths.extend([
                "C:\\Users\\%USERNAME%\\Videos",
                "D:\\Videos"
            ])
        else:  # Linux/macOS
            common_paths.extend([
                "/home/%USER%/Videos",
                "~/Videos"
            ])
        
        return common_paths
    
    def _get_common_output_paths(self):
        """Get list of common output directory paths."""
        common_paths = [
            "./output/",
            "./renders/",
            "./ComfyUI/output/",
            "./generated/"
        ]
        
        # Add platform-specific paths
        if os.name == 'nt':  # Windows
            common_paths.extend([
                "C:\\Users\\%USERNAME%\\Documents\\LoopyComfy",
                "D:\\Renders"
            ])
        else:  # Linux/macOS
            common_paths.extend([
                "/home/%USER%/Documents/LoopyComfy",
                "~/Documents/LoopyComfy"
            ])
        
        return common_paths


# Global instance
server_extensions = LoopyComfyServerExtensions()


def setup_routes(app):
    """
    Setup LoopyComfy routes with ComfyUI's web server.
    
    Args:
        app: ComfyUI web application instance
    """
    try:
        # Add all routes to the ComfyUI app
        for route in server_extensions.routes:
            app.router.add_route(route.method, route.path, route.handler)
        
        print("LoopyComfy: Server extensions registered successfully")
        
    except Exception as e:
        print(f"LoopyComfy: Failed to register server extensions: {str(e)}")


# ComfyUI integration - this will be called automatically if ComfyUI imports this module
try:
    # Try to import ComfyUI server components
    import server
    
    # Register routes with ComfyUI server
    if hasattr(server, 'PromptServer') and hasattr(server.PromptServer, 'instance'):
        setup_routes(server.PromptServer.instance.app)
        print("LoopyComfy: Integrated with ComfyUI PromptServer")
    else:
        print("LoopyComfy: ComfyUI PromptServer not available, routes will be registered later")
        
except ImportError:
    print("LoopyComfy: ComfyUI server not available, standalone mode")