from setuptools import setup

setup(
    name='fastMRI Prostate',
    version='1.0',
    description='A large scale dataset and reconstruction script of both raw prostate MRI measurements and images',
    install_requires=[
        'h5py==3.7.0',
        'numpy==1.23.5',
        'scikit-image==0.19.2'
    ],
    python_requires='>=3.9'
)