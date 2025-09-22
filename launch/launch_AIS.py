import json

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    completions = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nimbro_api'), '/launch/completions_launch.py']),
        # launch_arguments={
        #     # 'nimbro_api_completions_severity': "10",
        #     # 'nimbro_api_completions_nodes': "1",
        #     'nimbro_api_completions_api_endpoint': json.dumps({
        #         'name': "AIS",
        #         'api_flavor': "vllm",
        #         'models_url': "http://robo7:8000/v1/models",
        #         'completions_url': "http://robo7:8000/v1/chat/completions",
        #         'key_type': "environment",
        #         'key_value': "AIS_API_KEY"
        #         # 'key_value': "VLLM_API_KEY",
        #     }),
        #     'nimbro_api_completions_model_name': "ais/mimo-vl-7b-rl"
        #     # 'nimbro_api_completions_model_name': "rwth/mixtral-8x22B"
        # }.items()
    )

    embeddings = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nimbro_api'), '/launch/embeddings_launch.py']),
        launch_arguments={
            # 'nimbro_api_embeddings_severity': "10",
            # 'nimbro_api_embeddings_api_endpoint': json.dumps({
            #     'name': "AIS",
            #     'api_flavor': "openai",
            #     'models_url': "http://robo7:8000/v1/models",
            #     'embeddings_url': "https://robo7:8000/v1/embeddings",
            #     'key_type': "environment",
            #     'key_value': "AIS_API_KEY"
            # }),
            # 'nimbro_api_embeddings_model_name': "ais/embeddings"
        }.items()
    )

    images = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nimbro_api'), '/launch/images_launch.py'])
    )

    speech = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nimbro_api'), '/launch/speech_launch.py'])
    )

    nimbro_vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nimbro_api'), '/launch/nimbro_vision_launch.py']),
        launch_arguments={
            # 'nimbro_api_nimbro_vision_severity': "10",
            'nimbro_api_nimbro_vision_api_endpoint': "AIS",
            # 'nimbro_api_nimbro_vision_api_endpoint': json.dumps({
            #     'name': "AIS",

            #     'mmgroundingdino_url': "http://robo15:9035",
            #     'mmgroundingdino_key_type': "environment",
            #     'mmgroundingdino_key_value': "NIMBRO_VISION_API_KEY",

            #     'sam2_realtime_url': "http://robo15:9036",
            #     'sam2_realtime_key_type': "environment",
            #     'sam2_realtime_key_value': "NIMBRO_VISION_API_KEY",

            #     'dam_url': "http://robo15:9037",
            #     'dam_key_type': "environment",
            #     'dam_key_value': "NIMBRO_VISION_API_KEY",

            #     'florence2_url': "http://robo15:9038",
            #     'florence2_key_type': "environment",
            #     'florence2_key_value': "NIMBRO_VISION_API_KEY",

            #     'kosmos2_url': "http://robo15:9039",
            #     'kosmos2_key_type': "environment",
            #     'kosmos2_key_value': "NIMBRO_VISION_API_KEY",
            # })
        }.items()
    )

    return LaunchDescription([
        completions,
        embeddings,
        images,
        speech,
        nimbro_vision
    ])
