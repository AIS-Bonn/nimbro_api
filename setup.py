from glob import glob
from setuptools import setup, find_packages

package_name = 'nimbro_api'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(include=[f"{package_name}*"]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bastian Pätzold',
    maintainer_email='paetzold@ais.uni-bonn.de',
    description='This package exposes various APIs established by OpenAI (Chat Completions, Embeddings, Audio, Images) and adopted by others to ROS2.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'completions = nimbro_api.completions:main',
            'completions_multiplexer = nimbro_api.completions_multiplexer:main',
            'embeddings = nimbro_api.embeddings:main',
            'images = nimbro_api.images:main',
            'speech = nimbro_api.speech:main',
            'usage_monitor = nimbro_api.usage_monitor:main',
            'test = nimbro_api.utils.test:main',
            'toy_example_1 = nimbro_api.examples.toy_example_1:main',
            'toy_example_2 = nimbro_api.examples.toy_example_2:main',
            'toy_example_3 = nimbro_api.examples.toy_example_3:main',
            'toy_example_4 = nimbro_api.examples.toy_example_4:main'
        ]
    }
)
