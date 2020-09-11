from setuptools import setup, find_packages

setup(name='nipams_ml',
    version='0.1',
    # py_modules=['nipams_ml','nipams_ml.imports', 'nipams_ml.constants', 'nipams_ml.load', 'nipams_ml.experiments'],
    py_modules=['imports', 'constants', 'load', 'experiments'],
    packages=find_packages(),
)