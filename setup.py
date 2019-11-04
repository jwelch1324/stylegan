from setuptools import find_packages, setup

setup(
    name="menrva",
    packages=find_packages('src'),
    package_dir = {'':'src'},
    version='0.0.1',
    install_requires = ["numpy"],
    description='Art Generation Tools',
    author='Jonathan Welch',
)
