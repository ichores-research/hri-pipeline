from setuptools import setup, find_packages

package_name = "natural_language_processing"

setup(
    name=package_name,                 # ROS package name
    version="0.1.0",
    packages=find_packages(),          # <─ grabs *all* sub-packages
    install_requires=[
        "setuptools",
        "numpy",
        # add the real runtime deps here …
    ],
    data_files=[                       # keeps package.xml, resource index …
        ("share/" + package_name, ["package.xml"]),
    ],
    zip_safe=True,
    maintainer="petr",
    maintainer_email="petr.vanc@cvut.cz",
    description="Natural-language interface for HRI demo",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "nl_node = natural_language_processing.nl_node:main",
        ],
    },
)