"""
Legacy setup.py for backward compatibility.
Version is read from pyetc_wst/__init__.py — edit only that file to bump the version.
"""
from setuptools import setup
import re
import os

# Single source of truth for version
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "pyetc_wst", "__init__.py")) as _f:
    _version = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', _f.read(), re.M).group(1)

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pyetc_wst",
    version=_version,
    description="Exposure Time Calculator for the Wide-Field Spectroscopic Telescope (WST)",
    author="Matteo Ferro",
    author_email="matteo.ferro@inaf.it",
    url="https://github.com/ferromatteo/pyetc_wst",
    packages=["pyetc_wst"],
    package_data={
        "pyetc_wst": [
            "data/**/*",
            "Band_Filters/**/*.txt",
            "ESO_original_spectra/**/*",
        ]
    },
    use_scm_version=False,
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords="astronomy spectroscopy exposure-time-calculator WST",
)
