import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_prefix

launch_args = [
    DeclareLaunchArgument('nimbro_api_speech_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_speech_respawn_delay', default_value='1.0', description='Time in seconds waited before respawning nodes after a crash.'),
    DeclareLaunchArgument('nimbro_api_speech_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_speech_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_speech_api_endpoint', default_value='OpenAI', description="Sets the API endpoint defining API flavor, Models & Speech URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI']."),
    DeclareLaunchArgument('nimbro_api_speech_cache_read', default_value='True', choices=['True', 'False'], description='Attempt to retrieve speech from cached results.'),
    DeclareLaunchArgument('nimbro_api_speech_cache_write', default_value='True', choices=['True', 'False'], description='Cache retrieved speech locally.'),
    DeclareLaunchArgument('nimbro_api_speech_cache_folder', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "speech"), description='Path to the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_speech_cache_file', default_value='cache_speech.json', description='Name of the cache file inside the cache folder. If it does not exist it is automatically created.'),
]

def generate_launch_description():
    ld = LaunchDescription(launch_args)

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='speech',
                    name='speech',
                    namespace=context.launch_configurations['nimbro_api_speech_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_speech_respawn_delay']),
                    parameters=[
                        {
                            'severity': LaunchConfiguration('nimbro_api_speech_severity'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_speech_probe_api_connection'),
                            'api_endpoint': LaunchConfiguration('nimbro_api_speech_api_endpoint'),
                            'cache_read': LaunchConfiguration('nimbro_api_speech_cache_read'),
                            'cache_write': LaunchConfiguration('nimbro_api_speech_cache_write'),
                            'cache_folder': LaunchConfiguration('nimbro_api_speech_cache_folder'),
                            'cache_file': LaunchConfiguration('nimbro_api_speech_cache_file')
                        }
                    ]
                )
            ]
        )
    )

    return ld
