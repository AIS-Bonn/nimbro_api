import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_prefix
from launch_ros.parameter_descriptions import ParameterValue

launch_args = [
    DeclareLaunchArgument('nimbro_api_embeddings_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_embeddings_respawn_delay', default_value='1.0', description='Time in seconds waited before respawning nodes after a crash.'),
    DeclareLaunchArgument('nimbro_api_embeddings_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_embeddings_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_embeddings_api_endpoint', default_value='OpenAI', description="Sets the API endpoint defining API flavor, Models & Embeddings URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI', 'Mistral AI', 'vLLM', 'AIS']."),
    DeclareLaunchArgument('nimbro_api_embeddings_model_name', default_value='text-embedding-3-large', description='Name of the model that is used.'),
    DeclareLaunchArgument('nimbro_api_embeddings_cache_use', default_value='True', choices=['True', 'False'], description='Attempt to retrieve embeddings from cached results.'),
    DeclareLaunchArgument('nimbro_api_embeddings_cache_read_once', default_value='True', choices=['True', 'False'], description='Read embeddings cache file once when required and keep it in memory instead of loading it every time.'),
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
                            'severity': LaunchConfiguration('nimbro_api_embeddings_severity'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_embeddings_probe_api_connection'),
                            'api_endpoint': ParameterValue(LaunchConfiguration('nimbro_api_embeddings_api_endpoint'), value_type=str),
                            'model_name': LaunchConfiguration('nimbro_api_embeddings_model_name'),
                            'cache_use': LaunchConfiguration('nimbro_api_embeddings_cache_use'),
                            'cache_read_once': LaunchConfiguration('nimbro_api_embeddings_cache_read_once'),
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
