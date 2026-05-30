"""Setuptools entry point for editable installs.

The project metadata also lives in pyproject.toml.  This setup.py is kept as a
small compatibility shim for workflows or course tooling that still expect a
traditional setup.py file.
"""

from setuptools import find_packages, setup


setup(
    name="midi-error-detector",
    version="0.1.0",
    description="BiGRU wrong-note detection and correction experiments for MAESTRO MIDI.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "pretty_midi>=0.2.10",
        "numpy>=1.24",
        "pandas>=2.0",
        "tqdm>=4.66",
    ],
)
