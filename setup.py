from setuptools import setup, find_namespace_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = ["brainrender", "matplotlib", "numpy", "myterial", "rich"]

setup(
    name="bgheatmap",
    version="0.0.1",
    description="Brain regions heatmaps in brainrender",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    python_requires=">=3.6",
    packages=find_namespace_packages(exclude=("control")),
    include_package_data=True,
    url="",
    author="Federico Claudi",
    zip_safe=False,
    entry_points={"console_scripts": []},
)
