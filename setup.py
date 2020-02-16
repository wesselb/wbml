from setuptools import find_packages, setup

requirements = ['six',
                'numpy>=1.16',
                'pandas',
                'python-slugify',
                'requests',
                'netCDF4',

                'plum-dispatch',
                'backends>=0.3.1',
                'backends-matrix',
                'stheno',
                'varz>=0.3.0']

setup(packages=find_packages(exclude=['docs']),
      python_requires='>=3.6',
      install_requires=requirements,
      include_package_data=True)
