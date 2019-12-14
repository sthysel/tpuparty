# -*- encoding: utf-8 -*-
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='TPUParty',
    license='MIT',
    version='0.0.1',
    description='Tools and toys to use with the Coral TPU',
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'tpuparty-detection=tpuparty.detection:cli',
        ],
    },
    install_requires=[
        'click',
        'loguru',
        'opencv-python',
        'tensorflow',
    ],
    url='',
    classifiers=[
        'License :: MIT',
        'Development Status :: 4 - Beta',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
    keywords=[],
    extras_require={},
    setup_requires=[],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    package_data={
        '': ['config/*.yml'],
    },
)
