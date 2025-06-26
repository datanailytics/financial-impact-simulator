# Contributing to Data Analytics Portfolio

First off, thank you for considering contributing to this Data Analytics Portfolio! It's people like you that make this repository a great resource for the data science community. This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How Can I Contribute?](#how-can-i-contribute)
4. [Development Process](#development-process)
5. [Style Guidelines](#style-guidelines)
6. [Commit Guidelines](#commit-guidelines)
7. [Pull Request Process](#pull-request-process)
8. [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [your.email@example.com].

## Getting Started

### Prerequisites

Before you begin contributing, ensure you have the following installed:

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, or pipenv)
- Docker (optional, for containerized development)

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/data-analytics-portfolio.git
   cd data-analytics-portfolio
   ```

3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/original-owner/data-analytics-portfolio.git
   ```

4. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Additional development tools
   ```

6. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Exact steps to reproduce the problem
- Expected behavior vs actual behavior
- Screenshots (if applicable)
- Your environment details (OS, Python version, etc.)
- Any relevant log output

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Why this enhancement would be useful
- Possible implementation approaches
- Examples or mockups (if applicable)

### Contributing Code

#### Types of Contributions We're Looking For

1. **New Analysis Projects**: Complete data analysis projects with documentation
2. **Algorithm Implementations**: New machine learning or statistical algorithms
3. **Visualization Tools**: Enhanced data visualization capabilities
4. **Performance Improvements**: Optimizations to existing code
5. **Documentation**: Improvements to documentation, tutorials, or examples
6. **Bug Fixes**: Fixes for reported issues
7. **Tests**: Additional test coverage

#### Before Starting Work

1. Check if there's already an issue for what you want to work on
2. If not, create an issue to discuss your proposed changes
3. Wait for feedback before starting significant work
4. For small fixes, you can proceed directly to a pull request

### Contributing Data

If you're contributing datasets:

1. Ensure you have the right to share the data
2. Include proper documentation about the data source
3. Add a data dictionary explaining all features
4. Include any necessary citations or licenses
5. Ensure data is anonymized and contains no PII

## Development Process

### Branch Naming Convention

Use descriptive branch names following this pattern:
- `feature/descriptive-feature-name`
- `bugfix/issue-number-description`
- `docs/what-you-are-documenting`
- `refactor/what-you-are-refactoring`

### Development Workflow

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our style guidelines

3. Write or update tests as needed

4. Run tests locally:
   ```bash
   pytest tests/
   ```

5. Ensure code quality:
   ```bash
   black .
   flake8 .
   mypy .
   ```

6. Update documentation if needed

7. Commit your changes (see commit guidelines)

8. Push to your fork and create a pull request

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints where practical
- Write docstrings for all public functions and classes
- Use meaningful variable names

Example:
```python
def calculate_moving_average(
    data: pd.Series, 
    window: int = 7
) -> pd.Series:
    """
    Calculate moving average for a time series.
    
    Args:
        data: Time series data
        window: Window size for moving average
        
    Returns:
        Series with moving average values
    """
    return data.rolling(window=window).mean()
```

### Documentation Style

- Use Markdown for documentation files
- Include code examples where helpful
- Keep README files concise but comprehensive
- Update documentation alongside code changes

### Jupyter Notebook Guidelines

- Clear output before committing
- Include markdown cells explaining your analysis
- Use meaningful cell divisions
- Follow a logical flow: imports → data loading → EDA → modeling → results

## Commit Guidelines

### Commit Message Format

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(analysis): add customer segmentation using K-means

Implemented K-means clustering for customer segmentation with 
automatic elbow method for optimal cluster selection. Added 
visualization of clusters using PCA reduction.

Closes #123
```

### Commit Best Practices

- Make atomic commits (one feature/fix per commit)
- Write meaningful commit messages
- Reference issues in commit messages
- Keep the subject line under 50 characters
- Use the imperative mood in the subject line

## Pull Request Process

1. **Before Submitting**:
   - Ensure all tests pass
   - Update documentation
   - Add your changes to CHANGELOG.md
   - Rebase on the latest main branch

2. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] New tests added
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No sensitive data included
   ```

3. **Review Process**:
   - At least one maintainer review required
   - Address all review comments
   - Keep PR scope focused
   - Be responsive to feedback

4. **After Approval**:
   - Squash commits if requested
   - Ensure CI/CD passes
   - Maintainer will merge using squash-and-merge

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas
- **Email**: [your.email@example.com] for sensitive matters

### Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Annual contributor spotlight (for significant contributions)

### Resources for New Contributors

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [How to Write Good Commit Messages](https://chris.beams.io/posts/git-commit/)

## Questions?

Don't hesitate to ask questions! We were all beginners once, and we're here to help. The only bad question is the one not asked.

Thank you for contributing to making this portfolio a valuable resource for the data science community!