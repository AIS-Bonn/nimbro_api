import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from nimbro_utils.lazy import get_package_path

launch_args = [
    DeclareLaunchArgument('nimbro_api_images_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_images_respawn_delay', default_value='1.0', description='Time in seconds waited before respawning nodes after a crash.'),
    DeclareLaunchArgument('nimbro_api_images_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_images_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_images_api_endpoint', default_value='OpenAI', description="Sets the API endpoint defining API flavor, Models & Images URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI']."),
    DeclareLaunchArgument('nimbro_api_images_cache_read', default_value='True', choices=['True', 'False'], description='Attempt to retrieve images from cached results.'),
    DeclareLaunchArgument('nimbro_api_images_cache_write', default_value='True', choices=['True', 'False'], description='Cache retrieved images locally.'),
    DeclareLaunchArgument('nimbro_api_images_cache_folder', default_value=os.path.join(get_package_path("nimbro_api"), "cache", "images"), description='Path to the cache folder. If it does not exist it is automatically created.'),
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
                            'severity': LaunchConfiguration('nimbro_api_images_severity'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_images_probe_api_connection'),
                            'api_endpoint': LaunchConfiguration('nimbro_api_images_api_endpoint'),
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
