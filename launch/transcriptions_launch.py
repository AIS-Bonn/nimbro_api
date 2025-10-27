from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction

launch_args = [
    DeclareLaunchArgument('nimbro_api_transcriptions_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_transcriptions_respawn_delay', default_value='1.0', description='Time in seconds waited before respawning nodes after a crash.'),
    DeclareLaunchArgument('nimbro_api_transcriptions_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_transcriptions_log_line_length', default_value='150', description='Maximum line length of selected logger messages.'),
    DeclareLaunchArgument('nimbro_api_transcriptions_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the Models API of the endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_transcriptions_api_endpoint', default_value='OpenAI', description="Sets the API endpoint defining API flavor, Models & Transcriptions URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['OpenAI', 'vLLM', 'AIS'].")
]

def generate_launch_description():
    ld = LaunchDescription(launch_args)

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='transcriptions',
                    name='transcriptions',
                    namespace=context.launch_configurations['nimbro_api_transcriptions_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_transcriptions_respawn_delay']),
                    parameters=[
                        {
                            'severity': LaunchConfiguration('nimbro_api_transcriptions_severity'),
                            'log_line_length': LaunchConfiguration('nimbro_api_transcriptions_log_line_length'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_transcriptions_probe_api_connection'),
                            'api_endpoint': LaunchConfiguration('nimbro_api_transcriptions_api_endpoint')
                        }
                    ]
                )
            ]
        )
    )

    return ld
