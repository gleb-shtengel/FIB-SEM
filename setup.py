from pathlib import Path
from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()


setup(
    name="sift_gs",
    url="https://github.com/gleb-shtengel/FIB-SEM",
    author="Gleb Shtengel",
    description="Utilities for working with FIB-SEM data (Extract/analyze data, register FIB-SEM stacks, analyze noise etc.)",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["SIFT_gs"]),
    install_requires=[
        "dask",
        "ipython",
        "matplotlib",
        "mrcfile",
        "npy2bdv",
        "numpy",
        "opencv-contrib-python>3.4.1",
        "openpyxl",
        "pandas",
        "pillow",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "tifffile[all]",
        "tqdm",
    ],
    extras_require={"dev": ["flake8", "mypy"]},
    python_requires=">=3.8, <4.0",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
