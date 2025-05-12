from setuptools import setup, find_packages

setup(
    name="ibplusplus",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ibpp=IBplusplus.language:main"
        ]
    },
    install_requires=[],
    author="Your Name",
    description="IB++ Programming Language",
    python_requires=">=3.7",
)