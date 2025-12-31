from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['noetic_to_ros2'],
    package_dir={'': 'src'}
)

setup(**setup_args)

