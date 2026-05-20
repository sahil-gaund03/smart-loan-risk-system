from setuptools import setup, find_packages

setup(
    name="smart-loan-risk-system",
    version="1.0.0",
    author="Your Name",
    description="Production-grade AI/ML Loan Risk Prediction System",
    packages=find_packages(where="."),
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "train-loan-model=src.models.train:train",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
