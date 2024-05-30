from setuptools import setup, find_packages

setup(
    name='jimFisher',
    version='0.0.0.1',
    packages=find_packages(),
    install_requires=[
        'jimgw',
        'corner',
        'astropy',
        'tqdm'
        
    ],
    author='Jacob Golomb',
    author_email='jgolomb@caltech.edu',
    description='Generate samples. Do whatever you want with them.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jacobgolomb/jimFisher',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)