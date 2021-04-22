from setuptools import find_packages, setup

requirements = [
    "six",
    "numpy>=1.16",
    "pandas",
    "python-slugify",
    "requests",
    "netCDF4",
    "sklearn",
    "matplotlib",
    "xarray",
    "plum-dispatch>=1",
    "backends>=1",
    "backends-matrix>=1",
    "stheno>=1",
    "varz>=0.6",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
