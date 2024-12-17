from setuptools import find_packages, setup

requirements = [
    "six",
    "numpy>=1.16",
    "pandas",
    "pyarrow",
    "python-slugify",
    "requests",
    "netCDF4",
    "scikit-learn",
    "matplotlib",
    "xarray",
    "plum-dispatch>=2",
    "backends>=1",
    "backends-matrix>=1",
    "stheno>=1.4.2",
    "varz>=0.6",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
