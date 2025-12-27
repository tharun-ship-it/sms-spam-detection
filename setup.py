"""
SMS Spam Detection Package Setup

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sms-spam-detection",
    version="1.0.0",
    author="Tharun Ponnam",
    author_email="tharunponnam007@gmail.com",
    description="SMS Spam Detection using Machine Learning and NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tharun-ship-it/sms-spam-detection",
    project_urls={
        "Bug Tracker": "https://github.com/tharun-ship-it/sms-spam-detection/issues",
        "Documentation": "https://github.com/tharun-ship-it/sms-spam-detection#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "demo": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spam-detector=src.classifier:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "sms",
        "spam-detection",
        "machine-learning",
        "nlp",
        "text-classification",
        "nltk",
        "scikit-learn",
        "natural-language-processing",
    ],
)
