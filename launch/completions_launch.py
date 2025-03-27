import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_prefix

launch_args = [
    DeclareLaunchArgument('nimbro_api_completions_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_completions_respawn_delay', default_value='1.0', description='The delay in seconds after which a crashed node respawns.'),
    DeclareLaunchArgument('nimbro_api_completions_nodes', default_value='7', description='The number of completions nodes launched.'),
    DeclareLaunchArgument('nimbro_api_completions_logger_level', default_value='20', choices=['10', '20', '30', '40', '50'], description='Specifies the logging verbosity of all launched nodes (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50).'),
    DeclareLaunchArgument('nimbro_api_completions_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_completions_api_endpoint', default_value='OpenAI', description="Sets the API endpoint defining API flavor, Models & Completions URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI', 'Mistral AI', 'OpenRouter', 'AIS', 'vLLM']."),
    DeclareLaunchArgument('nimbro_api_completions_model_name', default_value='gpt-4o', description='Name of the model that is used.'),
    DeclareLaunchArgument('nimbro_api_completions_model_temperatur', default_value='1.0', description='Higher values like will make the output more random, while lower values like will make it more focused and deterministic.'),
    DeclareLaunchArgument('nimbro_api_completions_model_top_p', default_value='1.0', description='An alternative to sampling with temperature, called nucleus sampling, which behaves similar for similar values.'),
    DeclareLaunchArgument('nimbro_api_completions_model_max_tokens', default_value='1000', description='Maximum generated tokens per completion. Must be smaller than context size minus input tokens.'),
    DeclareLaunchArgument('nimbro_api_completions_model_presence_penalty', default_value='0.0', description='Positive values penalize new tokens based on whether they appear in the text so far.'),
    DeclareLaunchArgument('nimbro_api_completions_model_frequency_penalty', default_value='0.0', description='Positive values penalize new tokens based on their existing frequency in the text so far.'),
    DeclareLaunchArgument('nimbro_api_completions_stream_completion', default_value='True', choices=['True', 'False'], description='Using streaming to receive completions.'),
    DeclareLaunchArgument('nimbro_api_completions_normalize_text_response', default_value='True', choices=['True', 'False'], description='Applies text normalization to text responses (except JSON mode is used) without affecting the internal state of the message history.'),
    DeclareLaunchArgument('nimbro_api_completions_maximum_tool_calls_per_response', default_value='1', description="A response that is allowed to contain tool calls must contain at most this many tool calls. Set to '0' to deactivate."),
    DeclareLaunchArgument('nimbro_api_completions_correction_attempts', default_value='2', description='Invokes self-correction once invalid responses are being generated or timeouts take place.'),
    DeclareLaunchArgument('nimbro_api_completions_timeout_chunk', default_value='15.0', description='Time in seconds since last received chat completion chunk after which chat completion gets aborted to forward a invalid response.'),
    DeclareLaunchArgument('nimbro_api_completions_timeout_completion', default_value='60.0', description='Time in seconds since chat completion call after which chat completion gets aborted to forward a invalid response.'),
    DeclareLaunchArgument('nimbro_api_completions_timeout_service', default_value='5.0', description='Time in seconds after which service requests timeout that should respond immediately.'),
    DeclareLaunchArgument('nimbro_api_usage_logger_level', default_value='20', choices=['10', '20', '30', '40', '50'], description='Specifies the logging verbosity of all launched nodes (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50).'),
    DeclareLaunchArgument('nimbro_api_usage_cache_folder', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache"), description='Path to the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_file', default_value='cache_usage.json', description='Name of the cache file inside the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_read_once', default_value='True', choices=['True', 'False'], description='Read usage cache file once when required and keep it in memory instead of loading it every time.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_write_lazy', default_value='True', choices=['True', 'False'], description='Write usage cache file in fixed intervals instead of writing it with every update.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_write_interval', default_value='30.0', description='Minimum time in seconds in which the usage cache file is written if cache_write_lazy is active.')
]

def generate_launch_description():
    ld = LaunchDescription(launch_args)

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='completions',
                    name=f'completions_{i + 1}',
                    namespace=context.launch_configurations['nimbro_api_completions_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_completions_respawn_delay']),
                    parameters=[
                        {
                            'logger_level': LaunchConfiguration('nimbro_api_completions_logger_level'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_completions_probe_api_connection'),
                            'api_endpoint': LaunchConfiguration('nimbro_api_completions_api_endpoint'),
                            'model_name': LaunchConfiguration('nimbro_api_completions_model_name'),
                            'model_temperatur': LaunchConfiguration('nimbro_api_completions_model_temperatur'),
                            'model_top_p': LaunchConfiguration('nimbro_api_completions_model_top_p'),
                            'model_max_tokens': LaunchConfiguration('nimbro_api_completions_model_max_tokens'),
                            'model_presence_penalty': LaunchConfiguration('nimbro_api_completions_model_presence_penalty'),
                            'model_frequency_penalty': LaunchConfiguration('nimbro_api_completions_model_frequency_penalty'),
                            'stream_completion': LaunchConfiguration('nimbro_api_completions_stream_completion'),
                            'normalize_text_response': LaunchConfiguration('nimbro_api_completions_normalize_text_response'),
                            'maximum_tool_calls_per_response': LaunchConfiguration('nimbro_api_completions_maximum_tool_calls_per_response'),
                            'correction_attempts': LaunchConfiguration('nimbro_api_completions_correction_attempts'),
                            'timeout_chunk': LaunchConfiguration('nimbro_api_completions_timeout_chunk'),
                            'timeout_completion': LaunchConfiguration('nimbro_api_completions_timeout_completion')
                        }
                    ]
                ) for i in range(int(context.launch_configurations['nimbro_api_completions_nodes']))
            ]
        )
    )

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='completions_multiplexer',
                    name='completions_multiplexer',
                    namespace=context.launch_configurations['nimbro_api_completions_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_completions_respawn_delay']),
                    parameters=[
                        {
                            'logger_level': LaunchConfiguration('nimbro_api_completions_logger_level'),
                            'managed_nodes': [f"/{context.launch_configurations['nimbro_api_completions_namespace']}/completions_{i + 1}" for i in range(int(LaunchConfiguration('nimbro_api_completions_nodes').perform(context)))],
                            'timeout_completion': 300.0, # LaunchConfiguration('nimbro_api_completions_timeout_completion'),
                            'timeout_service': LaunchConfiguration('nimbro_api_completions_timeout_service')
                        }
                    ]
                ),
                Node(
                    package='nimbro_api',
                    executable='usage_monitor',
                    name='usage_monitor',
                    namespace=context.launch_configurations['nimbro_api_completions_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_completions_respawn_delay']),
                    parameters=[
                        {
                            'logger_level': LaunchConfiguration('nimbro_api_usage_logger_level'),
                            'cache_folder': LaunchConfiguration('nimbro_api_usage_cache_folder'),
                            'cache_file': LaunchConfiguration('nimbro_api_usage_cache_file'),
                            'cache_read_once': LaunchConfiguration('nimbro_api_usage_cache_read_once'),
                            'cache_write_lazy': LaunchConfiguration('nimbro_api_usage_cache_write_lazy'),
                            'cache_write_interval': LaunchConfiguration('nimbro_api_usage_cache_write_interval')
                        }
                    ]
                )
            ]
        )
    )

    return ld
