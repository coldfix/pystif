from setuptools import setup

setup(
    name='pystif',
    version='0.0.0',
    description='Shannon type inequality finder',
    long_description=None,
    author='Thomas Gläßle',
    author_email='t_glaessle@gmx.de',
    url=None,
    license='Public Domain',
    packages=['pystif'],
    install_requires=[
        'scipy',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    ],
)

