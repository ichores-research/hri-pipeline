from setuptools import find_packages, setup
from glob import glob

package_name = 'gesture_language_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        #('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    py_modules=[
        f"{package_name}.fusion_node",
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='michal.vavrecka@cvut.cz',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "fusion_node = gesture_language_fusion.fusion_node:main",
        ],
    },
)
