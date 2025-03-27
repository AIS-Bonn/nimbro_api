import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_prefix

launch_args = [
    DeclareLaunchArgument('nimbro_api_images_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_images_respawn_delay', default_value='1.0', description='The delay in seconds after which a crashed node respawns.'),
    DeclareLaunchArgument('nimbro_api_images_logger_level', default_value='20', choices=['10', '20', '30', '40', '50'], description='Specifies the logging verbosity of all launched nodes (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50).'),
    DeclareLaunchArgument('nimbro_api_images_api_flavor', default_value='openai', choices=['openai', 'mistral'], description='Sets API and accommodates for differences in their specifications.'),
    DeclareLaunchArgument('nimbro_api_images_cache_read', default_value='True', choices=['True', 'False'], description='Attempt to retrieve images from cached results.'),
    DeclareLaunchArgument('nimbro_api_images_cache_write', default_value='True', choices=['True', 'False'], description='Cache retrieved images locally.'),
    DeclareLaunchArgument('nimbro_api_images_cache_folder', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "images"), description='Path to the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_images_cache_file', default_value='cache_images.json', description='Name of the cache file inside the cache folder. If it does not exist it is automatically created.'),
]

def generate_launch_description():
    ld = LaunchDescription(launch_args)

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='images',
                    name='images',
                    namespace=context.launch_configurations['nimbro_api_images_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_images_respawn_delay']),
                    parameters=[
                        {
                            'logger_level': LaunchConfiguration('nimbro_api_images_logger_level'),
                            'api_flavor': LaunchConfiguration('nimbro_api_images_api_flavor'),
                            'cache_read': LaunchConfiguration('nimbro_api_images_cache_read'),
                            'cache_write': LaunchConfiguration('nimbro_api_images_cache_write'),
                            'cache_folder': LaunchConfiguration('nimbro_api_images_cache_folder'),
                            'cache_file': LaunchConfiguration('nimbro_api_images_cache_file')
                        }
                    ]
                )
            ]
        )
    )

    return ld
