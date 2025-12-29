from setuptools import find_packages, setup

# Read the contents of your requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="atlas",
    packages=find_packages(),
    install_requires=requirements,  # Add this line to include your requirements
)
