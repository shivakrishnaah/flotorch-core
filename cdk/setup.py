import setuptools

with open("README.md") as fp:
    long_description = fp.read()

setuptools.setup(
    name="Flotorch_infrastructure",
    version="0.1.0",
    description="Flotorch Infrastructure CDK Python app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="author",
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "aws-cdk-lib>=2.0.0",
        "constructs>=10.0.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
)