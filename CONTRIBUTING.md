# Contributing to IFUL

We welcome bug reports, feature requests, documentation improvements, and pull requests.

## Reporting Issues

If you encounter a bug or unexpected behavior:
1. Search existing GitHub Issues to see if it has already been reported.
2. If not, open a new **Bug Report** issue providing:
   - Python version and OS.
   - Minimal code snippet to reproduce the issue.
   - Full stack trace / error log.

## Proposing Features

For feature requests or enhancements:
1. Open a new **Feature Request** issue describing the proposed functionality and scientific use case.
2. Outline how the feature fits into `iful` (e.g. new kinematic profiles, datacube utilities, or solver improvements).

## Submitting Pull Requests

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/your-username/iful.git
   cd iful
   ```

2. **Set up Environment**:
   ```bash
   pip install -e .[test]
   ```

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

4. **Run Tests**:
   Ensure all existing and new unit tests pass cleanly:
   ```bash
   pytest -v --cov=iful
   ```

5. **Submit PR**:
   Push your branch to GitHub and open a Pull Request targeting `main`.

## Code Style Guidelines
- Follow PEP 8 guidelines.
- Use explicit relative imports within the `src/iful/` package.
- Include docstrings for new public functions and classes.
- Add test coverage in `tests/` for any new features or bug fixes.
