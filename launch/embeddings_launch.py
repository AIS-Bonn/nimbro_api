import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_prefix

launch_args = [
    DeclareLaunchArgument('nimbro_api_embeddings_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_embeddings_respawn_delay', default_value='1.0', description='The delay in seconds after which a crashed node respawns.'),
    DeclareLaunchArgument('nimbro_api_embeddings_logger_level', default_value='20', choices=['10', '20', '30', '40', '50'], description='Specifies the logging verbosity of all launched nodes (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50).'),
    DeclareLaunchArgument('nimbro_api_embeddings_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_embeddings_api_endpoint', default_value='OpenAI', description="Sets the API endpoint defining API flavor, Models & Completions URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI', 'Mistral AI', 'OpenRouter', 'AIS', 'vLLM']."),
    DeclareLaunchArgument('nimbro_api_embeddings_model_name', default_value='text-embedding-3-large', description='Name of the model that is used.'),
    DeclareLaunchArgument('nimbro_api_embeddings_cache_read', default_value='True', choices=['True', 'False'], description='Attempt to retrieve embeddings from cached results.'),
    DeclareLaunchArgument('nimbro_api_embeddings_cache_write', default_value='True', choices=['True', 'False'], description='Cache retrieved embeddings locally.'),
    DeclareLaunchArgument('nimbro_api_embeddings_cache_folder', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "embeddings"), description='Path to the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_embeddings_cache_file', default_value='cache_embeddings_index.json', description='Name of the cache file inside the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_embeddings_monitor_usage', default_value='True', choices=['True', 'False'], description='Tokenize input strings to monitor usage.')
]

def generate_launch_description():
    ld = LaunchDescription(launch_args)

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='embeddings',
                    name='embeddings',
                    namespace=context.launch_configurations['nimbro_api_embeddings_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_embeddings_respawn_delay']),
                    parameters=[
                        {
                            'logger_level': LaunchConfiguration('nimbro_api_embeddings_logger_level'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_embeddings_probe_api_connection'),
                            'api_endpoint': LaunchConfiguration('nimbro_api_embeddings_api_endpoint'),
                            'model_name': LaunchConfiguration('nimbro_api_embeddings_model_name'),
                            'cache_read': LaunchConfiguration('nimbro_api_embeddings_cache_read'),
                            'cache_write': LaunchConfiguration('nimbro_api_embeddings_cache_write'),
                            'cache_folder': LaunchConfiguration('nimbro_api_embeddings_cache_folder'),
                            'cache_file': LaunchConfiguration('nimbro_api_embeddings_cache_file'),
                            'monitor_usage': LaunchConfiguration('nimbro_api_embeddings_monitor_usage')
                        }
                    ]
                )
            ]
        )
    )

    return ld
