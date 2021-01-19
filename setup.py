import setuptools

with open("DESCRIPTION.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cost_based_selection",
    version="0.1",
    author="Louis Raynal, Jukka-Pekka Onnela",
    author_email="llcraynal@hsph.harvard.edu",
    description="A package containing implementations of various cost-based feature selection methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisRaynal/cost_based_selection",
    packages=setuptools.find_packages(include=['cost_based_selection',
                                               'cost_based_selection.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
            'joblib>=0.17.0',
            'matplotlib>=3.3.2',
            'networkx>=2.5',
            'numpy>=1.19.2',
            'pandas>=1.1.3',
            'rpy2>=2.9.3',
            'scipy>=1.5.2',
            'seaborn>=0.11.0',
            'sklearn'
    ],
    package_data={'': ['data/*.csv']}
)