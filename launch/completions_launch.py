import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_prefix
from launch_ros.parameter_descriptions import ParameterValue

launch_args = [
    DeclareLaunchArgument('nimbro_api_completions_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_completions_respawn_delay', default_value='1.0', description='Time in seconds waited before respawning nodes after a crash.'),
    DeclareLaunchArgument('nimbro_api_completions_nodes', default_value='7', description='The number of completions nodes launched.'),

    DeclareLaunchArgument('nimbro_api_completions_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_completions_log_line_length', default_value='150', description='Maximum line length of selected logger messages.'),
    DeclareLaunchArgument('nimbro_api_completions_log_last_messages', default_value='0', description='Number of newest messages in context logged with CompletionsPrompt request. Set -1 to log entire context.'),
    DeclareLaunchArgument('nimbro_api_completions_log_chunks', default_value='False', choices=['True', 'False'], description='Log all received chunks as DEBUG message.'),

    DeclareLaunchArgument('nimbro_api_completions_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_completions_api_endpoint', default_value='OpenRouter', description="Sets the API endpoint defining API flavor, Models & Completions URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI', 'Mistral AI', 'OpenRouter', 'vLLM', 'AIS']."),
    DeclareLaunchArgument('nimbro_api_completions_model_name', default_value='google/gemini-2.5-flash', description='Name of the model that is used.'),
    DeclareLaunchArgument('nimbro_api_completions_model_temperature', default_value='1.0', description='Higher values like will make the output more random, while lower values like will make it more focused and deterministic.'),
    DeclareLaunchArgument('nimbro_api_completions_model_top_p', default_value='1.0', description='An alternative to sampling with temperature, called nucleus sampling, which behaves similar for similar values.'),
    DeclareLaunchArgument('nimbro_api_completions_model_max_tokens', default_value='5000', description='Maximum number of tokens allowed to be generated for one Chat Completion.'),
    DeclareLaunchArgument('nimbro_api_completions_model_presence_penalty', default_value='0.0', description='Positive values penalize new tokens based on whether they appear in the text so far.'),
    DeclareLaunchArgument('nimbro_api_completions_model_frequency_penalty', default_value='0.0', description='Positive values penalize new tokens based on their existing frequency in the text so far.'),
    DeclareLaunchArgument('nimbro_api_completions_model_reasoning_effort', default_value='none', choices=['', 'none', 'low', 'medium', 'high'], description="Reasoning effort spent before generating the completion in ['', 'none', 'low', 'medium', 'high']."),
    DeclareLaunchArgument('nimbro_api_completions_completion_parsers', default_value='[""]', description='Define custom parsers to be executed in order after successful completions.'),
    DeclareLaunchArgument('nimbro_api_completions_completion_parsers_timeout', default_value='5.0', description='Time to wait in seconds for each completion parser to terminate.'),
    DeclareLaunchArgument('nimbro_api_completions_completion_parsers_folder', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "nimbro_api", "misc", "parsers", "completion"), description='Path to folder in which completion parsers are looked up first before interpreting them as global paths.'),
    DeclareLaunchArgument('nimbro_api_completions_stream_completion', default_value='True', choices=['True', 'False'], description='Using streaming to receive completions.'),
    DeclareLaunchArgument('nimbro_api_completions_normalize_text_response', default_value='False', choices=['True', 'False'], description='Applies text normalization to text responses (except JSON mode is used) without affecting the internal state of the context.'),
    DeclareLaunchArgument('nimbro_api_completions_maximum_tool_calls_per_response', default_value='1', description="A response that is allowed to contain tool calls must contain at most this many tool calls. Set to '0' to deactivate."),
    DeclareLaunchArgument('nimbro_api_completions_correction_attempts', default_value='0', description='Number of self-correction or retry attempts invoked after failed Chat Completions.'),
    DeclareLaunchArgument('nimbro_api_completions_timeout_chunk_first', default_value='10.0', description='Time in seconds waited until the next Chat Completion chunk is received.'),
    DeclareLaunchArgument('nimbro_api_completions_timeout_chunk_next', default_value='5.0', description='Time in seconds waited until the first Chat Completion chunk is received.'),
    DeclareLaunchArgument('nimbro_api_completions_timeout_completion', default_value='20.0', description='Time in seconds waited until a Chat Completion is finished.'),

    DeclareLaunchArgument('nimbro_api_multiplexer_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_multiplexer_timeout_service', default_value='5.0', description='Time in seconds waited for basic responses from service request.'),
    DeclareLaunchArgument('nimbro_api_multiplexer_timeout_completion', default_value='500.0', description='Time in seconds waited until a Chat Completion is finished.'),

    DeclareLaunchArgument('nimbro_api_usage_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_folder', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache"), description='Path to the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_file', default_value='cache_usage.json', description='Name of the cache file inside the cache folder. If it does not exist it is automatically created.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_read_once', default_value='True', choices=['True', 'False'], description='Read usage cache file once when required and keep it in memory instead of loading it every time.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_write_lazy', default_value='True', choices=['True', 'False'], description='Write usage cache file in fixed intervals instead of writing it with every update.'),
    DeclareLaunchArgument('nimbro_api_usage_cache_write_interval', default_value='30.0', description='Minimum time in seconds in which the usage cache file is written if cache_write_lazy is active.'),
    DeclareLaunchArgument('nimbro_api_usage_pricing_path', default_value=os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "nimbro_api", "misc", "pricing.json"), description='Path to the pricing file that stores the model cost per 1M tokens. Set empty string to disable price calculation.')
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
                            'severity': LaunchConfiguration('nimbro_api_completions_severity'),
                            'log_line_length': LaunchConfiguration('nimbro_api_completions_log_line_length'),
                            'log_last_messages': LaunchConfiguration('nimbro_api_completions_log_last_messages'),
                            'log_chunks': LaunchConfiguration('nimbro_api_completions_log_chunks'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_completions_probe_api_connection'),
                            'api_endpoint': ParameterValue(LaunchConfiguration('nimbro_api_completions_api_endpoint'), value_type=str),
                            'model_name': LaunchConfiguration('nimbro_api_completions_model_name'),
                            'model_temperature': LaunchConfiguration('nimbro_api_completions_model_temperature'),
                            'model_top_p': LaunchConfiguration('nimbro_api_completions_model_top_p'),
                            'model_max_tokens': LaunchConfiguration('nimbro_api_completions_model_max_tokens'),
                            'model_presence_penalty': LaunchConfiguration('nimbro_api_completions_model_presence_penalty'),
                            'model_frequency_penalty': LaunchConfiguration('nimbro_api_completions_model_frequency_penalty'),
                            'model_reasoning_effort': LaunchConfiguration('nimbro_api_completions_model_reasoning_effort'),
                            'completion_parsers': LaunchConfiguration('nimbro_api_completions_completion_parsers'),
                            'completion_parsers_timeout': LaunchConfiguration('nimbro_api_completions_completion_parsers_timeout'),
                            'completion_parsers_folder': LaunchConfiguration('nimbro_api_completions_completion_parsers_folder'),
                            'stream_completion': LaunchConfiguration('nimbro_api_completions_stream_completion'),
                            'normalize_text_response': LaunchConfiguration('nimbro_api_completions_normalize_text_response'),
                            'maximum_tool_calls_per_response': LaunchConfiguration('nimbro_api_completions_maximum_tool_calls_per_response'),
                            'correction_attempts': LaunchConfiguration('nimbro_api_completions_correction_attempts'),
                            'timeout_chunk_first': LaunchConfiguration('nimbro_api_completions_timeout_chunk_first'),
                            'timeout_chunk_next': LaunchConfiguration('nimbro_api_completions_timeout_chunk_next'),
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
                            'severity': LaunchConfiguration('nimbro_api_multiplexer_severity'),
                            'managed_nodes': [f"/{context.launch_configurations['nimbro_api_completions_namespace']}/completions_{i + 1}" for i in range(int(LaunchConfiguration('nimbro_api_completions_nodes').perform(context)))],
                            'timeout_service': LaunchConfiguration('nimbro_api_multiplexer_timeout_service'),
                            'timeout_completion': LaunchConfiguration('nimbro_api_multiplexer_timeout_completion')
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
                            'severity': LaunchConfiguration('nimbro_api_usage_severity'),
                            'cache_folder': LaunchConfiguration('nimbro_api_usage_cache_folder'),
                            'cache_file': LaunchConfiguration('nimbro_api_usage_cache_file'),
                            'cache_read_once': LaunchConfiguration('nimbro_api_usage_cache_read_once'),
                            'cache_write_lazy': LaunchConfiguration('nimbro_api_usage_cache_write_lazy'),
                            'cache_write_interval': LaunchConfiguration('nimbro_api_usage_cache_write_interval'),
                            'pricing_path': LaunchConfiguration('nimbro_api_usage_pricing_path')
                        }
                    ]
                )
            ]
        )
    )
    return ld
