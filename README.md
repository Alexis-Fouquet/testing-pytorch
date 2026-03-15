
This is a small project that I use to test the python library pytorch.

# Usage

To install required packages, install `nix` and use `nix develop -c bash`.
You can replace `bash` with any other shell (zsh, 42sh, ...).
It will also install python packages.

# Project tree

- `lib`: contains general classes and functions,
  without direct user interaction, like a lib.
- `src`: manage user interaction, might contain notebooks
- `tests`: contains unit and functional tests, if needed

# Installed python packages

- `torch` for Machine Learning
- `matplotlib` for visualisation
- `pytest` & `pytest-sugar` for unit tests
