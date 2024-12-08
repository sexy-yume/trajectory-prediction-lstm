from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    dev_requirements = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith("#") and not line.startswith("-r")
    ]

setup(
    name="trajectory-prediction-lstm",
    version="0.1.0",
    description="Trajectory prediction using LSTM with attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/trajectory-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    include_package_data=True,
    package_data={
        "trajectory_prediction": ["config/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "train-trajectory=src.train:main",
        ],
    }
)
