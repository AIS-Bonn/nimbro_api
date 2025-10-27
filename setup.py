from glob import glob
from setuptools import setup, find_packages

package_name = "nimbro_api"

setup(
    name=package_name,
    version="1.1.0",
    packages=find_packages(include=[f"{package_name}*", "examples*"]),
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
    description="Integration of various APIs (Chat Completions, Embeddings, Images, Speech, NimbRoVision) with ROS2.",
    license_files=["LICENSE"],
    entry_points={
        "console_scripts": [
            f"completions = {package_name}.completions:main",
            f"completions_multiplexer = {package_name}.completions_multiplexer:main",
            f"embeddings = {package_name}.embeddings:main",
            f"images = {package_name}.images:main",
            f"nimbro_vision = {package_name}.nimbro_vision:main",
            f"speech = {package_name}.speech:main",
            f"transcriptions = {package_name}.transcriptions:main",
            f"translations = {package_name}.translations:main",
            f"usage_monitor = {package_name}.usage_monitor:main",
            f"test = {package_name}.misc.test:main",
            "toy_example_1 = examples.toy_example_1:main",
            "toy_example_2 = examples.toy_example_2:main",
            "toy_example_3 = examples.toy_example_3:main",
            "toy_example_4 = examples.toy_example_4:main"
        ]
    }
)
