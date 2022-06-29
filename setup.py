from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "torch",
    "torchvision",
    "pyntcloud",
    "numpy",
    "matplotlib",
    "tqdm",
    "datetime",
    "umap-learn",
    "scikit-learn",
    "tensorboard",
]

setup(
    name="cellshape-cloud",
    version="0.0.15-rc0",
    description="3D cell shape analysis using geometric deep"
    " learning on point clouds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "black[jupyter]",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage>=5.0.3",
            "bump2version",
            "pre-commit",
            "flake8",
        ]
    },
    python_requires=">=3.7",
    packages=find_packages(),
    package_dir={"cellshape_cloud": "cellshape_cloud"},
    include_package_data=True,
    project_urls={
        "Source Code": "https://github.com/Sentinal4D/cellshape-cloud",
        "Bug Tracker": "https://github.com/Sentinal4D/cellshape-cloud/issues",
    },
    author="Matt De Vries, Lucas Dent, Adam Tyson",
    author_email="mattdevries.ai@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
