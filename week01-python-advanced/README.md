
# AI‑ML Journey — Week 01: Python (Advanced)

**Project Overview**

This small learning repository captures the first week's advanced Python work as part of an AI/ML journey. It contains focused exercises and simple implementations that demonstrate practical use of decorators, object-oriented design, data handling, and basic testing.

**Repository Structure**

- **`week01-python-advanced/main.py`**: project entry and examples.
- **`week01-python-advanced/decorators/timing.py`**: decorator examples for measuring execution time.
- **`week01-python-advanced/oops/`**: object-oriented examples and modules:
	- **`config.py`**, **`datasets.py`**, **`models.py`**, **`preprocessing.py`**
- **`test/test_basic.py`**: simple pytest tests covering core functions.

**What I Learned**

- **Decorators**: how to write and apply decorators, preserve function metadata, and use them for cross-cutting concerns (e.g., timing and logging). See [week01-python-advanced/decorators/timing.py](week01-python-advanced/decorators/timing.py#L1).
- **Object-Oriented Design**: organizing code into modules and classes for configuration, dataset handling, model interfaces, and preprocessing. This helped structure experiments and made code easier to test and extend. Key files: [week01-python-advanced/oops/config.py](week01-python-advanced/oops/config.py#L1), [week01-python-advanced/oops/models.py](week01-python-advanced/oops/models.py#L1).
- **Data Handling**: reading and preparing small datasets, implementing light preprocessing pipelines, and keeping transformation code modular and testable.
- **Testing with pytest**: writing simple unit tests to validate functions and components. Run the tests with `python -m pytest test/ -v`.
- **Project Layout & Best Practices**: separating exercises into folders, keeping a small and repeatable main entrypoint, and using tests to lock in expected behaviors.

**How to Run**

From the repository root:

```bash
cd week01-python-advanced
python main.py
```

Run tests:

```bash
python -m pytest test/ -v
```

**Key Takeaways**

- Start small: implement minimal, testable pieces (decorators, small classes) before combining them.
- Keep preprocessing separate from model code to make experiments reproducible.
- Use decorators for instrumentation (timing, logging) instead of scattering prints.
- Tests are lightweight safety nets — add them early.
