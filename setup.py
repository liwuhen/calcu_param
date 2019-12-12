#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=0.4.0+",
    "reduce"
]

setup(
    name='calcu_param',
    version='0.1.0',
    description="Calculate the parameters of the model",
    long_description=readme,
    author="lishaungshuang",
    author_email='578849202@qq.com',
    url='https://github.com/dicarlolab/CORnet',
    packages=['calcu_param'],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='kernel parameters calcu_param',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
)
