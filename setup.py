from distutils.core import setup

setup(
    name='lencoder',
    version='0.1.07',
    packages=['lencoder'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=["pyyaml", "numpy", "pandas"]
)
