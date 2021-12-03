import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoutils",
    version="0.0.0",
    author="Juan Antonio Barragan",
    author_email="jbarrag3@jh.edu",
    description="Surgical challenge solution code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "rich",
        "click",
    ],
    include_package_data=True,
    python_requires=">=3.7",
)
