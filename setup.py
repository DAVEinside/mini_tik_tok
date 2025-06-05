from setuptools import setup, find_packages

setup(
    name="video-recommender",
    version="1.0.0",
    author="nimit dave",
    description="Real-time video feed recommender system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "fastapi>=0.104.0",
        "redis>=5.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.4.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)