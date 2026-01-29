"""
Setup script untuk Diffusion Runtime package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "docs" / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Fault-Tolerant Diffusion Inference Runtime"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
    ]

setup(
    name="diffusion-runtime",
    version="1.0.0",
    author="Diffusion Runtime Team",
    description="Fault-tolerant diffusion inference runtime for memory-constrained environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
        "xformers": [
            "xformers>=0.0.20",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
