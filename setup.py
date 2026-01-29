from setuptools import setup, find_packages

setup(
    name="diffusion_runtime",
    version="1.0.0",
    description="SDFC: Stable Diffusion For Colab Core Engine",
    author="syapuanr",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy"
    ],
    python_requires=">=3.8",
)
