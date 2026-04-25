from setuptools import setup
import os
from glob import glob

package_name = 'robo_arp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'coordinator_node = robo_arp.coordinator_node:main',
            'slam_monitor_node = robo_arp.slam_monitor_node:main',
            'safety_monitor_node = robo_arp.safety_monitor_node:main',
            'wall_follower_wrapper_node = robo_arp.wall_follower_wrapper_node:main',
        ],
    },
)
