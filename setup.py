from setuptools import setup, find_packages

setup(
    name="JPAS_Domain_Adaptation",
    version="0.1.0",
    author="Daniel Lopez Cano",
    author_email="daniellopezcano13@gmail.com",
    description="Domain adaptation tools for JPAS object classification.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/daniellopezcano/JPAS_Domain_Adaptation",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.12.0",
        "matplotlib",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
