import os

from setuptools import setup, find_packages

name = 'od'
version = os.getenv('VERSION', '0.0.0')
description = 'Object Detection gRPC service'
url = 'https://github.com/arimanius/digivision-od'
author = 'Ahmad Salimi'
author_email = 'ahsa9978@gmail.com'
requirements_file = os.getenv('REQUIREMENTS_FILE', '../requirements.txt')


with open(requirements_file) as f:
    requirements = f.read().splitlines()

setup(
    name=name,
    version=version,
    description=description,

    url=url,
    author=author,
    author_email=author_email,

    packages=find_packages(),
    install_requires=requirements,

    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],

    entry_points={
        'console_scripts': [f'{name}={name}.cmd:main'],
    },
)
