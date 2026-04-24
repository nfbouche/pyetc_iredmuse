"""
Legacy setup.py for backward compatibility.
"""
from setuptools import setup

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pyetc_wst",
    version="1.1",
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
