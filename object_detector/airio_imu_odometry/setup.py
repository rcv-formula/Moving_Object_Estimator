from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'airio_imu_odometry'

setup(
    name=package_name,
    version='0.0.1',
    # ★ 서브패키지까지 포함
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='YOUR_EMAIL@example.com',
    description='AIR-IMU + AIR-IO odometry integration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'airio_node = airio_imu_odometry.airio_node:main',
            'imu_result_plot = airio_imu_odometry.nodes.imu_result_plot:main',
            'ate_rte_evaluator = airio_imu_odometry.nodes.ate_rte_evaluator:main',
            'profile_airio = airio_imu_odometry.nodes.profile_airio:main',
            'carto_airio_ekf_node = airio_imu_odometry.carto_airio_ekf_node:main',
            'carto_imu_ekf_node = airio_imu_odometry.carto_imu_ekf_node:main'
        ],
    },
)
