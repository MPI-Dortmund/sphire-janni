from setuptools import setup
import os
import re
import codecs
import janni
# Create new package with python setup.py sdist

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='janni',
    version=find_version("janni", "__init__.py"),
    python_requires='>3.4.0',
    packages=['janni'],
    url='https://github.com/MPI-Dortmund/sphire-janni',
    license='MIT',
    author='Thorsten Wagner',
    setup_requires=["Cython"],
    extras_require={
        'gpu': ['tensorflow-gpu == 1.10.1'],
        'cpu': ['tensorflow == 1.10.1']
    },
    install_requires=[
        "mrcfile >=1.0.0,<= 1.1.2",
        "Keras == 2.2.5",
        "numpy == 1.22.0",
        "h5py >= 2.5.0, < 3.0.0",
        "Pillow >= 6.0.0",
        "Cython",
        "imagecodecs-lite==2019.2.22",
        "tifffile==2019.7.2",
        "GooeyDev >= 1.0.3.2",
        "wxPython == 4.0.4",
        "scikit-image == 0.15.0"
    ],
    author_email='thorsten.wagner@mpi-dortmund.mpg.de',
    description='noise 2 noise for cryo em data',
    entry_points={
        'console_scripts': [
            'janni_denoise.py = janni.jmain:_main_'
        ]},
)
