"""Adapted from:
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='ufss',

    version='0.1.3',

    description='Package for simulating nonlinear optical spectra',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/peterarose/ufss/',

    author='Peter A. Rose and Jacob J. Krich',

    author_email='peter.rose56@gmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        
        'Topic :: Scientific/Engineering :: Physics',
        
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='nonlinear optical spectroscopy',

    packages=find_packages(exclude=['docs']),

    python_requires='>=3.5',

    install_requires=['numpy','matplotlib','pyfftw','scipy>=1','pyyaml','pyx'],

)
