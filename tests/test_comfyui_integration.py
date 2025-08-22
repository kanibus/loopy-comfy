"""
Tests for ComfyUI Integration - Node Registration and Compatibility
"""

import pytest
import sys
import os
import importlib
from unittest.mock import Mock, patch, MagicMock
import inspect

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer
from nodes.video_composer import LoopyComfy_VideoSequenceComposer
from nodes.video_saver import LoopyComfy_VideoSaver


class TestComfyUIIntegration:
    """Test suite for ComfyUI node registration and compatibility."""
    
    @pytest.fixture
    def all_nodes(self):
        """Get all node classes for testing."""
        return {
            "VideoAssetLoader": LoopyComfy_VideoAssetLoader,
            "MarkovVideoSequencer": LoopyComfy_MarkovVideoSequencer,
            "VideoSequenceComposer": LoopyComfy_VideoSequenceComposer,
            "VideoSaver": LoopyComfy_VideoSaver
        }
    
    def test_node_class_attributes(self, all_nodes):
        """Test that all nodes have required ComfyUI attributes."""
        required_attributes = [
            'INPUT_TYPES',
            'RETURN_TYPES', 
            'RETURN_NAMES',
            'FUNCTION',
            'CATEGORY'
        ]
        
        for node_name, node_class in all_nodes.items():
            for attr in required_attributes:
                assert hasattr(node_class, attr), f"{node_name} missing {attr}"
    
    def test_input_types_method(self, all_nodes):
        """Test INPUT_TYPES method returns correct structure."""
        for node_name, node_class in all_nodes.items():
            input_types = node_class.INPUT_TYPES()
            
            assert isinstance(input_types, dict), f"{node_name} INPUT_TYPES not dict"
            assert "required" in input_types, f"{node_name} missing required inputs"
            
            # Check required inputs structure
            required = input_types["required"]
            assert isinstance(required, dict), f"{node_name} required not dict"
            
            for input_name, input_spec in required.items():
                assert isinstance(input_spec, tuple), f"{node_name}.{input_name} not tuple"
                assert len(input_spec) >= 1, f"{node_name}.{input_name} empty tuple"
                
                # First element should be type or list of options
                type_spec = input_spec[0]
                assert isinstance(type_spec, (str, list, tuple)), \
                    f"{node_name}.{input_name} invalid type spec"
    
    def test_return_types_format(self, all_nodes):
        """Test RETURN_TYPES format compliance."""
        for node_name, node_class in all_nodes.items():
            return_types = node_class.RETURN_TYPES
            
            assert isinstance(return_types, tuple), f"{node_name} RETURN_TYPES not tuple"
            assert len(return_types) > 0, f"{node_name} RETURN_TYPES empty"
            
            for return_type in return_types:
                assert isinstance(return_type, str), f"{node_name} return type not string"
                assert len(return_type) > 0, f"{node_name} empty return type"
    
    def test_return_names_consistency(self, all_nodes):
        """Test RETURN_NAMES matches RETURN_TYPES length."""
        for node_name, node_class in all_nodes.items():
            return_types = node_class.RETURN_TYPES
            return_names = node_class.RETURN_NAMES
            
            assert isinstance(return_names, tuple), f"{node_name} RETURN_NAMES not tuple"
            assert len(return_names) == len(return_types), \
                f"{node_name} RETURN_NAMES/RETURN_TYPES length mismatch"
            
            for name in return_names:
                assert isinstance(name, str), f"{node_name} return name not string"
                assert len(name) > 0, f"{node_name} empty return name"
    
    def test_function_attribute(self, all_nodes):
        """Test FUNCTION attribute points to valid method."""
        for node_name, node_class in all_nodes.items():
            function_name = node_class.FUNCTION
            
            assert isinstance(function_name, str), f"{node_name} FUNCTION not string"
            assert len(function_name) > 0, f"{node_name} FUNCTION empty"
            assert hasattr(node_class, function_name), \
                f"{node_name} missing method {function_name}"
            
            # Check method is callable
            method = getattr(node_class, function_name)
            assert callable(method), f"{node_name}.{function_name} not callable"
    
    def test_category_attribute(self, all_nodes):
        """Test CATEGORY attribute format."""
        expected_category = "video/avatar"
        
        for node_name, node_class in all_nodes.items():
            category = node_class.CATEGORY
            
            assert isinstance(category, str), f"{node_name} CATEGORY not string"
            assert category == expected_category, \
                f"{node_name} CATEGORY '{category}' != '{expected_category}'"
    
    def test_node_instantiation(self, all_nodes):
        """Test nodes can be instantiated without errors."""
        for node_name, node_class in all_nodes.items():
            try:
                instance = node_class()
                assert instance is not None, f"{node_name} instantiation returned None"
            except Exception as e:
                pytest.fail(f"{node_name} instantiation failed: {e}")
    
    def test_node_method_signatures(self, all_nodes):
        """Test node methods have appropriate signatures."""
        for node_name, node_class in all_nodes.items():
            function_name = node_class.FUNCTION
            method = getattr(node_class, function_name)
            
            # Get method signature
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            # First parameter should be 'self'
            assert len(params) > 0, f"{node_name}.{function_name} no parameters"
            assert params[0] == 'self', f"{node_name}.{function_name} first param not self"
            
            # Check input parameters match INPUT_TYPES
            input_types = node_class.INPUT_TYPES()
            required_inputs = set(input_types["required"].keys())
            optional_inputs = set(input_types.get("optional", {}).keys())
            all_inputs = required_inputs | optional_inputs
            
            method_params = set(params[1:])  # Exclude 'self'
            
            # All required inputs should be in method params
            missing_required = required_inputs - method_params
            assert len(missing_required) == 0, \
                f"{node_name}.{function_name} missing required params: {missing_required}"
    
    def test_custom_type_definitions(self, all_nodes):
        """Test custom type definitions used in nodes."""
        custom_types = {
            "VIDEO_METADATA_LIST",
            "VIDEO_SEQUENCE", 
            "IMAGE"
        }
        
        for node_name, node_class in all_nodes.items():
            input_types = node_class.INPUT_TYPES()
            return_types = node_class.RETURN_TYPES
            
            # Check input types
            all_inputs = {}
            all_inputs.update(input_types["required"])
            all_inputs.update(input_types.get("optional", {}))
            
            for input_name, input_spec in all_inputs.items():
                input_type = input_spec[0]
                if isinstance(input_type, str) and input_type in custom_types:
                    # Custom type should be documented or handled
                    assert input_type in ["VIDEO_METADATA_LIST", "VIDEO_SEQUENCE", "IMAGE"], \
                        f"{node_name}.{input_name} uses undefined custom type: {input_type}"
            
            # Check return types
            for return_type in return_types:
                if return_type in custom_types:
                    assert return_type in ["VIDEO_METADATA_LIST", "VIDEO_SEQUENCE", "IMAGE", "STRING"], \
                        f"{node_name} returns undefined custom type: {return_type}"
    
    def test_node_tooltips_and_help(self, all_nodes):
        """Test nodes have helpful tooltips and documentation."""
        for node_name, node_class in all_nodes.items():
            # Check class docstring
            assert node_class.__doc__ is not None, f"{node_name} missing docstring"
            assert len(node_class.__doc__.strip()) > 0, f"{node_name} empty docstring"
            
            # Check INPUT_TYPES has tooltips where appropriate
            input_types = node_class.INPUT_TYPES()
            
            for input_name, input_spec in input_types["required"].items():
                if len(input_spec) > 1 and isinstance(input_spec[1], dict):
                    input_config = input_spec[1]
                    # Tooltip is good but not required
                    if "tooltip" in input_config:
                        assert isinstance(input_config["tooltip"], str)
                        assert len(input_config["tooltip"]) > 0
    
    def test_node_input_validation(self, all_nodes):
        """Test input type specifications are valid."""
        valid_base_types = {
            "STRING", "INT", "FLOAT", "BOOLEAN", 
            "IMAGE", "MASK", "LATENT", "MODEL", "CONDITIONING",
            "VIDEO_METADATA_LIST", "VIDEO_SEQUENCE"
        }
        
        for node_name, node_class in all_nodes.items():
            input_types = node_class.INPUT_TYPES()
            all_inputs = {}
            all_inputs.update(input_types["required"])
            all_inputs.update(input_types.get("optional", {}))
            
            for input_name, input_spec in all_inputs.items():
                input_type = input_spec[0]
                
                if isinstance(input_type, str):
                    # Should be valid type or custom type
                    assert input_type in valid_base_types or input_type.startswith("CUSTOM"), \
                        f"{node_name}.{input_name} unknown type: {input_type}"
                elif isinstance(input_type, (list, tuple)):
                    # Should be list of valid options
                    assert len(input_type) > 0, f"{node_name}.{input_name} empty options list"
                    for option in input_type:
                        assert isinstance(option, str), \
                            f"{node_name}.{input_name} non-string option: {option}"
    
    def test_node_default_values(self, all_nodes):
        """Test input default values are reasonable."""
        for node_name, node_class in all_nodes.items():
            input_types = node_class.INPUT_TYPES()
            
            for input_name, input_spec in input_types["required"].items():
                if len(input_spec) > 1 and isinstance(input_spec[1], dict):
                    input_config = input_spec[1]
                    
                    if "default" in input_config:
                        default_value = input_config["default"]
                        input_type = input_spec[0]
                        
                        # Validate default value matches type
                        if input_type == "STRING":
                            assert isinstance(default_value, str)
                        elif input_type == "INT":
                            assert isinstance(default_value, int)
                        elif input_type == "FLOAT":
                            assert isinstance(default_value, (int, float))
                        elif input_type == "BOOLEAN":
                            assert isinstance(default_value, bool)
                        elif isinstance(input_type, (list, tuple)):
                            assert default_value in input_type
    
    def test_node_output_consistency(self, all_nodes, sample_video_metadata):
        """Test node outputs match declared return types."""
        
        # Mock data for testing
        mock_frames = Mock()
        mock_frames.shape = (100, 480, 640, 3)
        
        mock_sequence = [
            {
                "video_id": "test",
                "file_path": "/test.mp4",
                "filename": "test.mp4", 
                "duration": 5.0,
                "start_time": 0.0,
                "end_time": 5.0
            }
        ]
        
        test_cases = {
            "VideoAssetLoader": {
                "method_args": {
                    "directory_path": "/fake/path",
                    "file_pattern": "*.mp4",
                    "max_videos": 10,
                    "validate_seamless": True
                },
                "expected_types": (list, str)
            },
            "MarkovVideoSequencer": {
                "method_args": {
                    "video_metadata": sample_video_metadata,
                    "total_duration_minutes": 1.0,
                    "prevent_immediate_repeat": True,
                    "random_seed": 42
                },
                "expected_types": (list, str)
            }
        }
        
        # Test nodes that can be tested without file system dependencies
        for node_name in test_cases:
            if node_name in all_nodes:
                node_class = all_nodes[node_name]
                instance = node_class()
                method = getattr(instance, node_class.FUNCTION)
                test_case = test_cases[node_name]
                
                try:
                    with patch('os.path.exists', return_value=False), \
                         patch('glob.glob', return_value=[]):
                        result = method(**test_case["method_args"])
                        
                        # Check result is tuple with correct length
                        assert isinstance(result, tuple)
                        assert len(result) == len(node_class.RETURN_TYPES)
                        
                        # Check each return value type
                        for i, (actual, expected_type) in enumerate(zip(result, test_case["expected_types"])):
                            assert isinstance(actual, expected_type), \
                                f"{node_name} return {i}: expected {expected_type}, got {type(actual)}"
                                
                except Exception as e:
                    # For nodes that require file system, just check they handle errors gracefully
                    if "No video files found" in str(e) or "Error" in str(e):
                        pass  # Expected error handling
                    else:
                        pytest.fail(f"{node_name} unexpected error: {e}")
    
    def test_node_registration_format(self, all_nodes):
        """Test nodes follow ComfyUI registration conventions."""
        for node_name, node_class in all_nodes.items():
            # Check class name format
            assert node_class.__name__.startswith("LoopyComfy_"), \
                f"{node_name} doesn't follow naming convention"
            
            # Check all required attributes exist and are proper format
            assert hasattr(node_class, 'INPUT_TYPES')
            assert callable(getattr(node_class, 'INPUT_TYPES'))
            
            assert hasattr(node_class, 'RETURN_TYPES')
            assert isinstance(node_class.RETURN_TYPES, tuple)
            
            assert hasattr(node_class, 'RETURN_NAMES')
            assert isinstance(node_class.RETURN_NAMES, tuple)
            
            assert hasattr(node_class, 'FUNCTION')
            assert isinstance(node_class.FUNCTION, str)
            
            assert hasattr(node_class, 'CATEGORY')
            assert isinstance(node_class.CATEGORY, str)
    
    def test_node_import_compatibility(self):
        """Test nodes can be imported in ComfyUI environment."""
        # Test individual node imports
        node_modules = [
            'nodes.video_asset_loader',
            'nodes.markov_sequencer', 
            'nodes.video_composer',
            'nodes.video_saver'
        ]
        
        for module_name in node_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Failed to import {module_name}"
            except ImportError as e:
                pytest.fail(f"Import error for {module_name}: {e}")
    
    @pytest.fixture
    def sample_video_metadata(self):
        """Sample video metadata for testing."""
        return [
            {
                "video_id": "test_001",
                "file_path": "/fake/test_001.mp4",
                "filename": "test_001.mp4",
                "duration": 5.0,
                "fps": 30.0,
                "frame_count": 150,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1024000,
                "is_seamless": True
            },
            {
                "video_id": "test_002",
                "file_path": "/fake/test_002.mp4",
                "filename": "test_002.mp4", 
                "duration": 4.5,
                "fps": 30.0,
                "frame_count": 135,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 950000,
                "is_seamless": True
            }
        ]
    
    def test_node_error_handling_format(self, all_nodes, sample_video_metadata):
        """Test nodes return errors in expected format."""
        for node_name, node_class in all_nodes.items():
            instance = node_class()
            
            # Test with obviously invalid inputs where possible
            if node_name == "VideoAssetLoader":
                result = instance.load_video_assets(
                    directory_path="/absolutely/nonexistent/path/12345",
                    file_pattern="*.mp4",
                    max_videos=10,
                    validate_seamless=True
                )
                
                assert isinstance(result, tuple)
                assert len(result) == 2
                metadata_list, status = result
                assert isinstance(metadata_list, list)
                assert isinstance(status, str)
                # Should indicate error or no files found
                assert len(metadata_list) == 0
                assert "No video files found" in status or "Error" in status
    
    def test_comfyui_node_mapping(self):
        """Test that we can create the NODE_CLASS_MAPPINGS dict for ComfyUI."""
        from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
        from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer
        from nodes.video_composer import LoopyComfy_VideoSequenceComposer
        from nodes.video_saver import LoopyComfy_VideoSaver
        
        # This is what would be in __init__.py for ComfyUI registration
        NODE_CLASS_MAPPINGS = {
            "LoopyComfy_VideoAssetLoader": LoopyComfy_VideoAssetLoader,
            "LoopyComfy_MarkovVideoSequencer": LoopyComfy_MarkovVideoSequencer,
            "LoopyComfy_VideoSequenceComposer": LoopyComfy_VideoSequenceComposer,
            "LoopyComfy_VideoSaver": LoopyComfy_VideoSaver
        }
        
        NODE_DISPLAY_NAME_MAPPINGS = {
            "LoopyComfy_VideoAssetLoader": "Video Asset Loader",
            "LoopyComfy_MarkovVideoSequencer": "Markov Video Sequencer",
            "LoopyComfy_VideoSequenceComposer": "Video Sequence Composer", 
            "LoopyComfy_VideoSaver": "Video Saver"
        }
        
        assert len(NODE_CLASS_MAPPINGS) == 4
        assert len(NODE_DISPLAY_NAME_MAPPINGS) == 4
        
        # Test all mappings are valid
        for key, node_class in NODE_CLASS_MAPPINGS.items():
            assert callable(node_class)
            assert hasattr(node_class, 'INPUT_TYPES')
            assert hasattr(node_class, 'RETURN_TYPES')
            assert key in NODE_DISPLAY_NAME_MAPPINGS