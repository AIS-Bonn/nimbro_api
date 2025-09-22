from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.parameter_descriptions import ParameterValue

launch_args = [
    DeclareLaunchArgument('nimbro_api_nimbro_vision_namespace', default_value='nimbro_api', description='The namespace of all launched nodes.'),
    DeclareLaunchArgument('nimbro_api_nimbro_vision_respawn_delay', default_value='1.0', description='Time in seconds waited before respawning nodes after a crash.'),
    DeclareLaunchArgument('nimbro_api_nimbro_vision_severity', default_value='20', choices=['10', '20', '30', '40', '50'], description='Logging severity of node logger.'),
    DeclareLaunchArgument('nimbro_api_nimbro_vision_probe_api_connection', default_value='True', choices=['True', 'False'], description='Probes the API endpoint to validate the API key and model name.'),
    DeclareLaunchArgument('nimbro_api_nimbro_vision_probe_model_state', default_value='True', choices=['True', 'False'], description='Probes the model state before inference and loads the requested model if required.'),
    DeclareLaunchArgument('nimbro_api_nimbro_vision_api_endpoint', default_value='localhost', description="Sets the API endpoint defining URLs, key type and value. Must be a valid JSON encoded dictionary or a name in ['localhost', 'AIS']."),
]

def generate_launch_description():
    ld = LaunchDescription(launch_args)

    ld.add_action(
        OpaqueFunction(
            function=lambda context: [
                Node(
                    package='nimbro_api',
                    executable='nimbro_vision',
                    name='nimbro_vision',
                    namespace=context.launch_configurations['nimbro_api_nimbro_vision_namespace'],
                    output='full',
                    emulate_tty=True,
                    respawn=True,
                    respawn_delay=float(context.launch_configurations['nimbro_api_nimbro_vision_respawn_delay']),
                    parameters=[
                        {
                            'severity': LaunchConfiguration('nimbro_api_nimbro_vision_severity'),
                            'probe_api_connection': LaunchConfiguration('nimbro_api_nimbro_vision_probe_api_connection'),
                            'probe_model_state': LaunchConfiguration('nimbro_api_nimbro_vision_probe_model_state'),
                            'api_endpoint': ParameterValue(LaunchConfiguration('nimbro_api_nimbro_vision_api_endpoint'), value_type=str)
                        }
                    ]
                )
            ]
        )
    )

    return ld
