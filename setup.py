#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py
"""
Setup files.

Copyright (c) 2024, gleb.shtengel@gmail.com
"""

import setuptools
import codecs
import os.path

def get_version(rel_path):
     with codecs.open(rel_path, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")

version=get_version("SIFT_gs/__init__.py")

setuptools.setup(
    name="FIBSEM_gs",
    version=version,
    author="Gleb Shtengel",
    author_email="gleb.shtengel@gmail.com",
    url="https://github.com/gleb-shtengel/FIB-SEM",
    description="Python library for processing and analysis of FIB-SEM data",
    long_description=open('README.md', "r").read(),
    long_description_content_type="text/markdown",
    license="BSD 3",
    packages=setuptools.find_packages(exclude=("tests", "tests.*")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=[line.strip() for line in open('requirements.txt', "r")],
)