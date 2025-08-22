# Contributing to ComfyUI NonLinear Video Avatar

Thank you for your interest in contributing to ComfyUI NonLinear Video Avatar! We welcome contributions from the community and are grateful for your help in making this project better.

## 🚀 Quick Start

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/loopy-comfy.git
   cd loopy-comfy
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

6. **Run tests to ensure everything works**:
   ```bash
   pytest tests/ -v
   ```

## 🎯 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug reports** - Help us identify and fix issues
- **Feature requests** - Suggest new functionality
- **Code contributions** - Fix bugs or implement features
- **Documentation** - Improve guides, examples, and API docs
- **Testing** - Add test cases and improve coverage
- **Performance optimization** - Make the code faster and more efficient

### Reporting Issues

When reporting issues, please:

1. **Use the issue templates** when available
2. **Include reproduction steps** with minimal example
3. **Provide system information**:
   - OS and version
   - Python version
   - ComfyUI version
   - GPU information (if applicable)
4. **Include error logs** and stack traces
5. **Check existing issues** to avoid duplicates

### Submitting Pull Requests

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Run the test suite**:
   ```bash
   pytest tests/ -v --cov=core --cov=nodes --cov=utils
   ```

5. **Run code quality checks**:
   ```bash
   black nodes/ core/ utils/ --check
   pylint nodes/ core/ utils/
   mypy nodes/ core/ utils/
   ```

6. **Commit your changes** with descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add visual similarity transition mode"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## 📋 Coding Standards

### Code Style

We use **Black** for code formatting with these settings:
- Line length: 88 characters
- String quotes: Double quotes preferred
- Format on save recommended

```bash
# Format all code
black nodes/ core/ utils/ tests/

# Check formatting without changes
black nodes/ core/ utils/ tests/ --check
```

### Linting

We use **Pylint** for code quality:
- Target score: 9.0+
- Configuration in `.pylintrc`

```bash
pylint nodes/ core/ utils/ --rcfile=.pylintrc
```

### Type Hints

We use **MyPy** for type checking:
- All public functions must have type hints
- Configuration in `mypy.ini`

```bash
mypy nodes/ core/ utils/ --config-file mypy.ini
```

### Import Organization

We use **isort** for import sorting:
- Standard library first
- Third-party packages second
- Local imports last

```bash
isort nodes/ core/ utils/ tests/
```

## 🧪 Testing Guidelines

### Test Requirements

- **New features** must include comprehensive tests
- **Bug fixes** should include regression tests
- **Target coverage**: 90%+ for new code
- **Performance tests** for optimization changes

### Test Structure

```python
# tests/test_new_feature.py
import pytest
from unittest.mock import Mock, patch
from core.new_feature import NewFeature

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic feature operation."""
        feature = NewFeature()
        result = feature.process()
        assert result is not None
    
    def test_edge_case_handling(self):
        """Test edge case behavior."""
        feature = NewFeature()
        with pytest.raises(ValueError):
            feature.process(invalid_input=True)
    
    @patch('core.new_feature.external_dependency')
    def test_mocked_dependencies(self, mock_dep):
        """Test with mocked external dependencies."""
        mock_dep.return_value = "expected_value"
        feature = NewFeature()
        result = feature.process()
        assert result == "expected_value"
```

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific test file
pytest tests/test_markov_engine.py -v

# With coverage
pytest tests/ --cov=core --cov=nodes --cov=utils --cov-report=html

# Performance tests
pytest tests/test_performance.py --benchmark-only
```

## 📚 Documentation

### Docstring Style

We use **Google-style docstrings**:

```python
def generate_sequence(
    self, 
    duration: float, 
    seed: Optional[int] = None
) -> List[MarkovState]:
    """
    Generate a non-repetitive video sequence.
    
    Args:
        duration: Target duration in seconds
        seed: Random seed for reproducible results
        
    Returns:
        List of MarkovState objects representing the sequence
        
    Raises:
        ValueError: If duration is negative or zero
        RuntimeError: If no valid states available
        
    Example:
        >>> engine = MarkovTransitionEngine(states)
        >>> sequence = engine.generate_sequence(60.0, seed=123)
        >>> len(sequence) > 0
        True
    """
    pass
```

### README Updates

When adding features:
- Update feature list in README.md
- Add usage examples
- Update installation instructions if needed
- Include troubleshooting information

## 🏗️ Architecture Guidelines

### Core Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Use dependency injection for testability
3. **Error Handling**: Comprehensive error handling with meaningful messages
4. **Performance**: Consider memory usage and processing speed
5. **Extensibility**: Design for future enhancements

### File Organization

```
ComfyUI-NonLinearVideoAvatar/
├── nodes/              # ComfyUI node implementations
├── core/               # Core business logic
├── utils/              # Shared utilities
├── tests/              # Test suite
├── docs/               # Documentation
├── workflows/          # Example workflows
└── assets/             # Sample assets (excluded from git)
```

### Adding New Features

When adding new features:

1. **Design first**: Create issue with design proposal
2. **Core logic**: Implement in `core/` module
3. **ComfyUI integration**: Add node in `nodes/`
4. **Utilities**: Add helpers in `utils/` if needed
5. **Tests**: Comprehensive test coverage
6. **Documentation**: Update README and add docstrings

## 🐛 Debugging

### Common Issues

**Import Errors**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python path configuration

**Test Failures**:
- Run tests individually to isolate issues
- Check for missing test dependencies
- Verify mock configurations

**Performance Issues**:
- Profile code with `py-spy` or `line_profiler`
- Monitor memory usage with `memory_profiler`
- Use smaller test datasets for debugging

### Debug Configuration

Add this to your IDE for debugging:
```json
{
    "name": "Debug Tests",
    "type": "python",
    "request": "launch",
    "module": "pytest",
    "args": ["tests/test_specific.py::test_function", "-v", "-s"],
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}"
}
```

## 📊 Performance Considerations

### Memory Management

- Use batch processing for large datasets
- Implement proper cleanup in `__del__` methods
- Monitor peak memory usage during development
- Test with memory-constrained environments

### Processing Speed

- Profile critical paths with timing decorators
- Use numpy vectorization where possible
- Consider parallel processing for independent operations
- Benchmark against performance targets

### Quality Assurance

- Maintain frame-perfect transitions
- Validate output quality with metrics (PSNR, SSIM)
- Test with various video formats and resolutions
- Ensure consistent behavior across platforms

## 🚀 Release Process

### Version Management

We use **Semantic Versioning**:
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR  
- Bug fixes increment PATCH

### Release Checklist

1. **Update version** in `__init__.py`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** and ensure all tests pass
4. **Update documentation** as needed
5. **Create release** on GitHub with detailed notes

## 🎯 Development Priorities

### Current Focus Areas

1. **Core Stability** - Bug fixes and edge case handling
2. **Performance Optimization** - Speed and memory improvements
3. **Test Coverage** - Comprehensive test suite expansion
4. **Documentation** - User guides and API documentation
5. **ComfyUI Integration** - Enhanced workflow capabilities

### Future Roadmap

1. **Visual Similarity** - Advanced transition modes
2. **External Inputs** - Sensor integration capabilities
3. **ML Integration** - Machine learning-based transitions
4. **Real-time Processing** - Live streaming capabilities

## 🤝 Community

### Communication

- **GitHub Issues** for bug reports and feature requests
- **GitHub Discussions** for questions and ideas
- **Pull Requests** for code contributions
- **Code Review** process for quality assurance

### Code of Conduct

We follow a code of conduct based on respect and collaboration:
- Be welcoming to newcomers
- Be respectful in all interactions
- Focus on constructive feedback
- Help others learn and grow

## 📞 Getting Help

If you need help:

1. **Check documentation** first
2. **Search existing issues** on GitHub
3. **Create new issue** with detailed information
4. **Join discussions** for broader questions

Thank you for contributing to ComfyUI NonLinear Video Avatar! Your contributions help make digital avatars more lifelike and engaging for everyone.