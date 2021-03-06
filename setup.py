# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Predict sales prices on the Kaggle House Prices dataset using DVC for reproducible machine learning',
    url='https://github.com/truocpham-agilityio/kaggle-house-prices-dvc',
    author='Truoc Pham',
    author_email='info@teamio.com',
    license='MIT',
    zip_safe=False,
    keyword=['pip', 'houseprices']
)
