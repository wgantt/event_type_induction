# UDS Event Type Induction

This package contains code for a generative event type induction model over
the Universal Decompositional Semantics corpus.

## Usage

Training, evaluation, and debugging is done through scripts in the `scripts`
directory, and the way that module imports are structured assumes this setup.
Scripts must be run from the root (i.e. `code`) directory. To run, enter:

```python -m scripts.{script_name} [parameters]```
