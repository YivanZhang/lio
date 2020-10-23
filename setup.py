import pathlib
from setuptools import setup, find_packages

NAME = 'lio'
VERSION = '0.1.0'
DESCRIPTION = 'Learning from Indirect Observations'

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

AUTHOR = 'Yivan Zhang'
EMAIL = 'yivanzhang@ms.k.u-tokyo.ac.jp'
URL = 'https://github.com/YivanZhang/lio'

PYTHON_REQUIRES = '>=3.8.0'
PACKAGES = find_packages()
INSTALL_REQUIRES = [
    'numpy',
    'torch',
    'torchvision',
]

setup(
    # meta
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    # author
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    # package
    packages=PACKAGES,
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    # PyPI
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)
