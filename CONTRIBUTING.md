# Contributing to Blood-Based Cancer Mathematical Model

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the blood-based cancer model repository.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [your.email@domain.com].

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear, descriptive title
- Detailed description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, package versions)
- Code samples or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- Clear description of the enhancement
- Rationale and use cases
- Potential implementation approach (if you have ideas)
- Any relevant research papers or references

### Contributing Code

1. **Fork the repository** and create a feature branch
2. **Make your changes** with clear, focused commits
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/blood-cancer-model.git
cd blood-cancer-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### Development Dependencies

```bash
# Install development tools
pip install pytest pytest-cov black flake8 mypy sphinx
```

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- Maximum line length: 100 characters
- Use Black for code formatting
- Use type hints for function signatures
- Write docstrings for all public functions

### Code Formatting

```bash
# Format code with Black
black src/

# Check style with flake8
flake8 src/ --max-line-length=100

# Type checking with mypy
mypy src/
```

### Documentation Style

- Use Google-style docstrings
- Include parameter types and return types
- Provide examples for complex functions
- Document mathematical equations in LaTeX format

Example:
```python
def predict_treatment_response(
    biomarkers: Dict[str, float],
    treatment_protocol: str
) -> Tuple[float, float]:
    """
    Predict treatment response from biomarker panel.
    
    Args:
        biomarkers: Dictionary mapping biomarker names to values
        treatment_protocol: Name of treatment protocol ('hormone', 'chemo', etc.)
    
    Returns:
        Tuple of (response_probability, confidence_score)
        
    Raises:
        ValueError: If biomarkers are out of valid range
        
    Example:
        >>> biomarkers = {'ca153': 45.0, 'cea': 5.2, 'cd8': 850}
        >>> response, confidence = predict_treatment_response(biomarkers, 'hormone')
        >>> print(f"Response: {response:.2f}, Confidence: {confidence:.2f}")
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_mathematical_validation.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

- Write unit tests for all new functions
- Aim for >80% code coverage
- Include edge cases and error conditions
- Use descriptive test names

Example test:
```python
import pytest
from src.production_system import BloodPanel

def test_blood_panel_validation():
    """Test that blood panel validates biomarker ranges correctly."""
    # Valid panel
    panel = BloodPanel(ca153=45.0, cea=5.2)
    errors, warnings = panel.validate()
    assert len(errors) == 0
    
    # Invalid panel (negative value)
    panel_invalid = BloodPanel(ca153=-10.0)
    errors, warnings = panel_invalid.validate()
    assert len(errors) > 0
```

## Documentation

### Building Documentation

```bash
cd docs/
make html  # Generate HTML documentation
make pdf   # Generate PDF documentation
```

### Documentation Structure

- **README.md**: Overview and quick start
- **docs/**: Detailed technical documentation
- **notebooks/**: Tutorial Jupyter notebooks
- **Code comments**: Inline explanations for complex logic

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
   
   Use clear commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Code style changes
   - `perf:` Performance improvements

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Provide clear title and description
   - Reference any related issues
   - Describe testing performed
   - Include screenshots for UI changes

### PR Review Process

- Maintainers will review within 1-2 weeks
- Address review comments promptly
- Keep PRs focused and reasonably sized
- CI tests must pass before merging

## Areas for Contribution

We especially welcome contributions in:

- **Additional ML models**: Implementing new machine learning architectures
- **Clinical validation**: Real-world data analysis and validation
- **Visualization**: Enhanced plotting and interactive visualizations
- **Performance optimization**: Speed and memory improvements
- **Documentation**: Tutorials, examples, and guides
- **Testing**: Expanding test coverage
- **Feature engineering**: Novel biomarker combinations
- **Multi-cancer extension**: Extending to other cancer types

## Questions?

Feel free to:
- Open an issue for questions
- Email the maintainers at [your.email@domain.com]
- Join our discussion forum [if available]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to advancing precision oncology through mathematical modeling!
