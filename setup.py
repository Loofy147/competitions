from setuptools import setup, find_packages

setup(
    name="competition_ai",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
        "bitsandbytes",
    ],
)
