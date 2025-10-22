from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mujoco_arm_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 👇 add this line to include your launch files
        (os.path.join('share', package_name, 'launch'), glob('mujoco_arm_demo/launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yahia',
    maintainer_email='yahiaamr90@gmail.com',
    description='Simple Mujoco arm demo integrated with ROS 2',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        "console_scripts": [
            "simple_move_node = mujoco_arm_demo.simple_move_node:main",
        ],
    },
)
