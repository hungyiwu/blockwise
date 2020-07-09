import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='blockwise',
    version='0.1',
    author='Hung-Yi Wu',
    author_email='hungyiwu@protonmail.com',
    description='Fast blockwise operations of array data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hungyiwu/blockwise',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'dask[array]',
        ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        ],
    )
