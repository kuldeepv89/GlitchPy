from setuptools import setup

py_modules = ["main", "plots", "supportGlitch", "utils_general"]
setup(
    name="GlitchPy",
    version="0.1",
    py_modules=py_modules,
    install_requires=[ "h5py==3.13.0", 
        "scipy==1.15.2", 
        "scikit-learn==1.6.1", 
        "matplotlib==3.10.1", 
        "seaborn==0.13.2", 
        "numpy==2.2.3", 
        "meson==1.8.2", 
        "ninja==1.11.1" ],
    entry_points={"console_scripts": ["GlitchPyrun=main:main"]},
)

