from setuptools import setup
import os
import re
import codecs
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

def get_tensorflow(*file_paths):
    version_file = read(*file_paths)
    if "dev0"in version_file:
        print("Use tensorflow CPU")
        return "tensorflow == 1.12.3"
    else:
        print("Use tensorflow GPU")
        return "tensorflow-gpu == 1.12.3"
setup(
    name='janni',
    version=find_version("janni", "__init__.py"),
    python_requires='>3.4.0',
    packages=['janni'],
    url='https://github.com/MPI-Dortmund/sphire-janni',
    license='MIT',
    author='Thorsten Wagner',
    setup_requires=["Cython"],
    install_requires=[
        "mrcfile >= 1.0.0",
        "Keras >= 2.2.4",
        "numpy == 1.15.4",
        "h5py >= 2.5.0",
        "Cython",
        "imagecodecs-lite==2019.2.22",
        "tifffile",
        get_tensorflow("janni", "__init__.py"),
    ],
    author_email='thorsten.wagner@mpi-dortmund.mpg.de',
    description='noise 2 noise for cryo em data',
    entry_points={
        'console_scripts': [
            'janni_denoise.py = janni.jmain:_main_'
        ]},
)
