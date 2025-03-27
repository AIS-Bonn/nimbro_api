#!/usr/bin/env python3

import os
import json
import time
import datetime

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange
from ament_index_python.packages import get_package_prefix

from nimbro_api_interfaces.srv import GetSpeech
from nimbro_api.utils.node import start_and_spin_node
from nimbro_api.utils.parameter_handler import ParameterHandler

### <Parameter Defaults>

node_name = "speech"
logger_level = 10
api_flavor = "openai"

cache_read = True
cache_write = True
cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "speech")
cache_file = "cache_speech.json"

## non-params

api_settings = { # TODO switch to endpoints concept as in completions/embeddings
    'openai': {
        'key_variable': "OPENAI_API_KEY",
        'models_url': "https://api.openai.com/v1/models",
        'speech_url': "https://api.openai.com/v1/audio/speech"
    }
}

### </Parameter Defaults>

class Speech(Node):

    def __init__(self):
        super().__init__(node_name)
        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self.parameter_handler = ParameterHandler(self)
        self.add_on_set_parameters_callback(self.parameter_handler.parameter_callback)

        descriptor = ParameterDescriptor()
        descriptor.name = "logger_level"
        descriptor.type = ParameterType.PARAMETER_INTEGER
        descriptor.description = "Logger level of this node (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50)."
        descriptor.read_only = False
        int_range = IntegerRange()
        int_range.from_value = 10
        int_range.to_value = 50
        int_range.step = 10
        descriptor.integer_range.append(int_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, logger_level, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "api_flavor"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = f"Sets API and accommodates for differences in their specifications. Must be in {list(api_settings.keys())}."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, api_flavor, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_read"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Attempt to retrieve speech from cached results."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_read, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_write"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Cache retrieved speech locally."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_write, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_folder"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Path to the cache folder. If it does not exist it is automatically created."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_folder, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_file"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Name of the cache file inside the cache folder. If it does not exist it is automatically created."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_file, descriptor)

        self.parameter_handler.all_declared()

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)

        self.cbg_speech = ReentrantCallbackGroup()
        self.srv_speech = self.create_service(GetSpeech, f"{self.node_namespace}/{self.node_name}/get_speech".replace("//", "/"), self.get_speech_callack, qos_profile=qos_profile, callback_group=self.cbg_speech)

        self.get_logger().info("Node started")

    def __del__(self):
        self.get_logger().info("Node shutdown")

    def parameter_changed(self, parameter):
        success = True
        reason = ""

        if parameter.name == "logger_level":
            self.logger_level = parameter.value
            rclpy.logging.set_logger_level(f"{self.node_namespace}/{self.node_name}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(self.logger_level))

        elif parameter.name == "api_flavor":
            success, reason = self.connect_api(parameter.value)

        elif parameter.name == "cache_read":
            self.cache_read = parameter.value

        elif parameter.name == "cache_write":
            if not self.cache_read and parameter.value is True:
                self.get_logger().warn("Activating 'cache_read' in order to activate 'cache_write'")
                results = self.set_parameters([rclpy.parameter.Parameter("cache_write", type_=rclpy.parameter.Parameter.Type(1), value=True)])
                success = results[0].successful
                reason = results[0].reason
                if success:
                    self.cache_write = parameter.value
            else:
                self.cache_write = parameter.value

        elif parameter.name == "cache_folder":
            if parameter.value == "":
                self.cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache")
                self._node.get_logger().info(f"Interpreting empty parameter 'cache_folder' as '{self.cache_folder}'")
            else:
                self.cache_folder = parameter.value

        elif parameter.name == "cache_file":
            self.cache_file = parameter.value

        else:
            return None, None

        return success, reason

    def connect_api(self, api_flavor):
        api_flavor = api_flavor.lower()

        if hasattr(self, api_flavor):
            if self.api_flavor == api_flavor:
                success = True
                message = f"Already connected to '{api_flavor}' API"
                return success, message

        if api_flavor not in list(api_settings.keys()):
            success = False
            message = f"Value '{api_flavor}' is not in list of available API flavors {list(api_settings.keys())}"
        else:
            self.get_logger().debug(f"Connecting to '{api_flavor}' API")

            success = True
            message = f"Successfully connected to '{api_flavor}' API"

            api_key = os.getenv(api_settings[api_flavor]['key_variable'])
            if api_key is None:
                success = False
                message = f"Could not find API key. Get one from '{api_flavor}' and 'echo \"export {api_settings[api_flavor]['key_variable']}='yourkey'\" >> ~/.bashrc'"
            else:
                self.get_logger().debug(f"Using API key '{api_key}'")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                try:
                    available_models = requests.get(api_settings[api_flavor]['models_url'], headers=headers)
                except Exception as e:
                    success = False
                    message = f"Failed to retrieve available models: {e}"
                else:
                    if available_models.status_code == 200:
                        available_models = [m['id'] for m in available_models.json()['data']]
                        if len(available_models) == 0:
                            success = False
                            message = "There are no models available under this API"
                        else:
                            self.get_logger().debug(f"Available models: {available_models}")
                    else:
                        success = False
                        message = f"Failed to retrieve available models: {available_models.text}"

        if success:
            self.api_flavor = api_flavor
            self.api_key = api_key
            self.available_models = available_models

        return success, message

    # Speech Pipeline

    def speech_post(self, text, model, voice, speed, api_url, api_key):
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }

        data = {
            'input': text,
            'model': model,
            'voice': voice,
            'speed': speed,
            'response_format': "wav"
        }

        # TODO add timeout
        # try:
        #     requests.post(url, data=payload, timeout=5)
        # except requests.Timeout:
        #     # back off and retry
        #     pass
        # except requests.ConnectionError:
        #     pass

        self.get_logger().debug(f"Posting request: {data}")
        tic = time.perf_counter()
        response = requests.post(api_url, headers=headers, json=data, stream=False)
        toc = time.perf_counter()
        self.get_logger().debug(f"Received response after '{toc - tic:.3f}s'")

        if response.status_code == 200:
            success = True
            message = "Successfully retrieved speech."
            speech_bytes = response.content
        else:
            success = False
            message = f"HTTP-Error: {response.text}"
            speech_bytes = None

        return success, message, speech_bytes

    def get_speech(self, text, model, voice, speed):
        # parse arguments

        # TODO normalize prompt

        if text == "":
            message = "Cannot generate speech for empty text."
            self.get_logger().error(message)
            return False, message, None

        supported_models = ["tts-1", "tts-1-hd"]
        if model == "":
            model = supported_models[0]
            self.get_logger().debug(f"Using default model '{model}'")

        if model not in supported_models:
            message = f"Model '{model}' is not supported. Supported models are: {supported_models}"
            self.get_logger().error(message)
            return False, message, None

        if model not in self.available_models:
            message = f"Model '{model}' is not available. Available models are: {self.available_models}"
            self.get_logger().error(message)
            return False, message, None

        supported_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice == "":
            voice = supported_voices[0]
            self.get_logger().debug(f"Using default voice '{voice}'")

        if voice not in supported_voices:
            message = f"Voice '{voice}' is not supported. Supported voices are: {supported_voices}"
            self.get_logger().error(message)
            return False, message, None

        speed_range = [0.25, 4.0]
        if speed < speed_range[0] or speed > speed_range[1]:
            self.get_logger().warn(f"Clipping speed '{speed}' to interval {speed_range}")
            speed = max(speed_range[0], min(speed_range[1], speed))

        # read cache

        speech_path = None

        cache_read, cache_write = self.cache_read, self.cache_write

        if cache_read:
            # check if cache file exists
            cache_path = os.path.join(self.cache_folder, self.cache_file)
            if not os.path.isfile(cache_path):
                cache = {}
                self.get_logger().warn(f"Cache file '{cache_path}' doesn't exist")
            else:
                # open file
                self.get_logger().debug(f"Reading cache file '{cache_path}'")
                try:
                    with open(cache_path, 'r') as f:
                        cache = json.load(f)
                except Exception as e:
                    self.get_logger().warn(f"Failed to open cache file '{cache_path}': {e}")
                else:
                    speech_path = cache.get(model, {}).get(voice, {}).get(str(speed), {}).get(text)
                    if speech_path is None:
                        self.get_logger().debug("Speech not found in cache")
                    else:
                        self.get_logger().debug(f"Found speech in cache '{speech_path}'")

        # generate speech if necessary

        if speech_path is None:
            # post to API

            self.get_logger().debug(f"Retrieving speech from API (text='{text}', model='{model}', voice='{voice}', speed='{speed}')")
            success, message, speech_bytes = self.speech_post(
                text=text,
                model=model,
                voice=voice,
                speed=speed,
                api_url=api_settings[api_flavor]['speech_url'],
                api_key=self.api_key
            )
            if not success:
                self.get_logger().error(message)
                return False, message, None

            # create cache folder

            if not os.path.exists(self.cache_folder):
                self.get_logger().debug(f"Creating cache folder '{self.cache_folder}'")
                try:
                    os.makedirs(self.cache_folder)
                except Exception as e:
                    self.get_logger().error(f"Failed to create cache folder '{self.cache_folder}': {e}")

            # write speech to file

            stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            speech_path = os.path.join(self.cache_folder, f"{stamp}.wav")
            self.get_logger().debug(f"Writing speech to file '{speech_path}'")

            try:
                with open(speech_path, mode='bw') as f:
                    f.write(speech_bytes)
            except Exception as e:
                message = f"Failed to write speech to file '{speech_path}': {e}"
                self.get_logger().error(message)
                return False, message, None

            # write path to cache

            if cache_write:
                cache_path = os.path.join(self.cache_folder, self.cache_file)
                self.get_logger().debug(f"Writing speech path to cache file '{cache_path}'")

                # add path to cache

                if model not in cache:
                    cache[model] = {}
                if voice not in cache[model]:
                    cache[model][voice] = {}
                if str(speed) not in cache[model][voice]:
                    cache[model][voice][str(speed)] = {}
                cache[model][voice][str(speed)][text] = speech_path

                # write cache

                if os.path.exists(self.cache_folder):
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump(cache, f, indent=4)
                    except Exception as e:
                        self.get_logger().error(f"Failed to save speech path to cache file '{cache_path}': {e}")

        # forward results

        self.get_logger().info(f"Successfully retrieved speech '{speech_path}' (text='{text}', model='{model}', voice='{voice}', speed='{speed}')")

        return True, "Successfully retrieved speech.", speech_path

    # Callbacks

    def get_speech_callack(self, request, response):
        self.get_logger().debug("get_speech_callack(): start")

        response.success, response.message, speech_path = self.get_speech(
            text=request.text,
            model=request.model,
            voice=request.voice,
            speed=request.speed
        )
        if response.success:
            response.path = speech_path

        self.get_logger().debug("get_speech_callack(): end")
        return response

def main(args=None):
    start_and_spin_node(Speech, args=args)

if __name__ == '__main__':
    main()
