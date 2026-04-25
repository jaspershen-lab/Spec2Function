from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="spec2function",
    version="0.1.3",
    description="Deep learning model for MS2 spectrum annotation and metabolite set analysis",
    author="Feifan Zhang",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://huggingface.co/cgxjdzz/ms2function-assets",

    packages=find_packages(exclude=("tests", "examples", "dist")),
    include_package_data=True,

    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.35",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
        "huggingface_hub>=0.20",
        "python-dotenv>=1.0",
        "openai>=1.30",
    ],
    extras_require={
        "train": ["wandb>=0.16"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
