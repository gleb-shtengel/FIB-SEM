#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py
"""
Setup files.

Copyright (c) 2023, Gleb Shtengel
"""

import os
import setuptools
import versioneer

setuptools.setup(
    name="FIBSEM_gs",
    version='4.0.1',
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
    python_requires=">=3.8",
    install_requires=[line.strip() for line in open('requirements.txt', "r")],
)