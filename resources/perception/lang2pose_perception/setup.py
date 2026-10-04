from setuptools import setup

package_name = "lang2pose_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hyeonsu Oh",
    maintainer_email="hans324oh@gmail.com",
    description="FoundationPose 6D pose estimation for Lang2Pose",
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "pose_estimator = lang2pose_perception.pose_estimator:main",
        ],
    },
)
