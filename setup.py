#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/2 15:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='perovskite',
    version='0.0.01',
    keywords=['perovskite'," structure"],
    description='This is one tool to generator perovskite structures for calculation, such as First-principles calculations.',
    install_requires=['pandas', 'numpy', 'scipy', 'joblib', 'matplotlib', 'deprecated',
                      'requests', 'tqdm', "mgetool", "pymatgen", "numba", "ase"],
    include_package_data=True,
    author='wangchangxin',
    author_email='986798607@qq.com',
    python_requires='>=3.6',
    url='https://github.com/boliqq07/perovskite',
    maintainer='wangchangxin',
    platforms=[
        "Windows",
        "Unix",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],

    packages=find_packages(
        exclude=["test", "*.test", "*.*.test", "*.*.*.test",
                 "test*", "*.test*", "*.*.test*", "*.*.*.test*", "Instances", "Instance*"],
    ),
    long_description=long_description,
    long_description_content_type='text/markdown'
)
