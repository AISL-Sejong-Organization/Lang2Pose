from setuptools import setup
import os
from glob import glob

package_name = 'aiagent'

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        # Include package.xml
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("lib/python3.10/site-packages/aiagent", ["../../.env"]),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="Hyeonsu Oh",
    maintainer_email="hans324oh@gmail.com",
    description="A ROS 2 package that uses ChatGPT to act as a robot agent.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realrobot = aiagent.realrobot:main",
            "simrobot = aiagent.simrobot:main",
        ],
    },
)
