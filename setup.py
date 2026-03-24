from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="spec2function",
    version="0.1.1",
    description="Deep learning model for MS2 data annotation",
    author="User",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://huggingface.co/cgxjdzz/ms2function-assets",

    # 閾ｪ蜉ｨ譟･謇ｾ蠖灘燕逶ｮ蠖穂ｸ狗噪蛹?(霑咎㈹蜿ｪ莨壽伽蛻ｰ MS2BioText)
    packages=find_packages(),

    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "wandb",
        "huggingface_hub",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
