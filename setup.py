from glob import glob
from setuptools import setup, find_packages

package_name = "nimbro_api"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(include=[f"{package_name}*"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py"))
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bastian Pätzold",
    maintainer_email="paetzold@ais.uni-bonn.de",
    description="This package exposes various APIs (Chat Completions, Embeddings, Images, Speech, NimbRoVision) to ROS2.",
    license_files=["LICENSE"],
    entry_points={
        "console_scripts": [
            f"completions = {package_name}.completions:main",
            f"completions_multiplexer = {package_name}.completions_multiplexer:main",
            f"embeddings = {package_name}.embeddings:main",
            f"images = {package_name}.images:main",
            f"nimbro_vision = {package_name}.nimbro_vision:main",
            f"speech = {package_name}.speech:main",
            f"usage_monitor = {package_name}.usage_monitor:main",
            f"test = {package_name}.utils.test:main",
            f"toy_example_1 = {package_name}.examples.toy_example_1:main",
            f"toy_example_2 = {package_name}.examples.toy_example_2:main",
            f"toy_example_3 = {package_name}.examples.toy_example_3:main",
            f"toy_example_4 = {package_name}.examples.toy_example_4:main"
        ]
    }
)
