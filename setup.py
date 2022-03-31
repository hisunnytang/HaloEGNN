from setuptools import setup, find_packages

setup(
    name = 'HaloFlows',
    version='0.0.1',
    packages=find_packages(exclude=['flow_model','train_script']),

)

