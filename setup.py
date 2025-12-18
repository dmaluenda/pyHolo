from setuptools import setup, find_packages

setup(
    name='pyHolo',
    version='0.1.0',
    long_description=open('README.md').read(),
    description='Program to generate holograms for laser beam engineering.',
    url='https://github.com/dmaluenda/pyHolo',
    author='David Maluenda',
    author_email='dmaluenda@ub.edu',
    license='GNU Public License v3',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering'
    ],
)