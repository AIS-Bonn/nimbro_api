#!/usr/bin/env python3

import os
import json
import time
import datetime

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from nimbro_api_interfaces.srv import SpeechGet
from nimbro_api.misc.common import validate_default_endpopints, filter_api_endpoint, validate_api_endpoint, retrieve_api_key, probe_models_api, validate_connection

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, get_package_path

### <Parameter Defaults>

node_name = "speech"
severity = 10

probe_api_connection = True
api_endpoint = "OpenAI"

cache_read = True
cache_write = True
cache_folder = os.path.join(get_package_path("nimbro_api"), "cache", "speech")
cache_file = "cache_speech.json"

## non-params

line_length = 150

api_endpoints = {
    'OpenAI': {
        'api_flavor': "openai",
        'models_url': "https://api.openai.com/v1/models",
        'speech_url': "https://api.openai.com/v1/audio/speech",
        'key_type': "environment",
        'key_value': "OPENAI_API_KEY"
    }
}

### </Parameter Defaults>

class Speech(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # initialize endpoints

        self.endpoint_keys_required = {'name', 'api_flavor', 'speech_url', 'key_type', 'key_value'}
        self.endpoint_keys_optional = {'models_url'}
        self.endpoint_key_type_values = ["environment", "plain"]
        self.endpoint_api_flavor_values = ["openai"]
        validate_default_endpopints.__get__(self)(api_endpoints)

        self.filter_api_endpoint = filter_api_endpoint.__get__(self)
        self.validate_api_endpoint = validate_api_endpoint.__get__(self)
        self.retrieve_api_key = retrieve_api_key.__get__(self)
        self.probe_models_api = probe_models_api.__get__(self)
        self.validate_connection = validate_connection.__get__(self)

        self.api_endpoints = api_endpoints
        self.endpoint_probes = {}

        # declare parameters

        self.parameter_handler = ParameterHandler(self)

        self.parameter_handler.declare(
            name="severity",
            dtype=int,
            default_value=severity,
            description="Logging severity of node logger.",
            read_only=False,
            range_min=10,
            range_max=50,
            range_step=10
        )

        self.parameter_handler.declare(
            name="probe_api_connection",
            dtype=bool,
            default_value=probe_api_connection,
            description="Probes the Models API of the endpoint to validate the API key and model name.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="api_endpoint",
            dtype=str,
            default_value=api_endpoint,
            description=f"Sets the API endpoint defining API flavor, Models & Speech URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_read",
            dtype=bool,
            default_value=cache_read,
            description="Attempt to retrieve speech from cached results.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_write",
            dtype=bool,
            default_value=cache_write,
            description="Cache retrieved speech locally.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_folder",
            dtype=str,
            default_value=cache_folder,
            description="Path to the cache folder. If it does not exist it is automatically created.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_file",
            dtype=str,
            default_value=cache_file,
            description="Name of the cache file inside the cache folder. If it does not exist it is automatically created.",
            read_only=False
        )

        # create interfaces

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)

        self.cbg_speech = ReentrantCallbackGroup()
        self.srv_speech = self.create_service(SpeechGet, f"{self.node_namespace}/{self.node_name}/get_speech".replace("//", "/"), self.get_speech_callack, qos_profile=qos_profile, callback_group=self.cbg_speech)

        self._logger.info("Node started")

    def __del__(self):
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self._logger.set_settings(settings={'severity': value})

        elif name == "probe_api_connection":
            if is_declared and value != self.parameters.probe_api_connection:
                self._logger.debug("Reset endpoint probes")
                self.endpoint_probes = {}

        elif name == "api_endpoint":
            value, message = self.filter_api_endpoint(name, value, line_length)

        elif name == "cache_write":
            if not self.parameters.cache_read and value is True:
                self._logger.warn("Activating 'cache_read' in order to activate 'cache_write'")
                results = self.set_parameters([rclpy.parameter.Parameter("cache_write", type_=rclpy.parameter.Parameter.Type(1), value=True)])
                success = results[0].successful
                if not success:
                    message = results[0].reason
                    value = None

        elif name == "cache_folder":
            if value == "":
                value = os.path.join(get_package_path("nimbro_api"), "cache")

        return value, message

    # Speech Pipeline

    def speech_post(self, text, model, voice, speed, instructions, api_url, api_key):
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

        if instructions != "":
            data['instructions'] = instructions

        # TODO add timeout
        # try:
        #     requests.post(url, data=payload, timeout=5)
        # except requests.Timeout:
        #     # back off and retry
        #     pass
        # except requests.ConnectionError:
        #     pass

        self._logger.debug(f"Posting request: {data}")
        tic = time.perf_counter()
        response = requests.post(api_url, headers=headers, json=data, stream=False)
        toc = time.perf_counter()
        self._logger.debug(f"Received response after '{toc - tic:.3f}s'")

        if response.status_code == 200:
            success = True
            message = "Retrieved speech."
            speech_bytes = response.content
        else:
            success = False
            message = f"HTTP-Error: {response.text}"
            speech_bytes = None

        return success, message, speech_bytes

    def get_speech(self, text, model, voice, speed, instructions):
        # parse arguments

        if text == "":
            message = "Cannot generate speech for empty text."
            self._logger.error(message)
            return False, message, None

        supported_models = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
        if model == "":
            model = supported_models[0]
            self._logger.debug(f"Using default model '{model}'")

        if instructions != "" and model != "gpt-4o-mini-tts":
            message = f"Model '{model}' does not support instructions."
            self._logger.error(message)
            return False, message, None

        if model not in supported_models:
            message = f"Model '{model}' is not supported. Supported models are: {supported_models}"
            self._logger.error(message)
            return False, message, None

        supported_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
        if voice == "":
            voice = supported_voices[0]
            self._logger.debug(f"Using default voice '{voice}'")

        if voice not in supported_voices:
            message = f"Voice '{voice}' is not supported. Supported voices are: {supported_voices}"
            self._logger.error(message)
            return False, message, None

        speed_range = [0.25, 4.0]
        if speed < speed_range[0] or speed > speed_range[1]:
            self._logger.warn(f"Clipping speed '{speed}' to interval {speed_range}")
            speed = max(speed_range[0], min(speed_range[1], speed))

        # read cache

        speech_path = None

        cache_read, cache_write = self.parameters.cache_read, self.parameters.cache_write

        if cache_read:
            # check if cache file exists
            cache_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)
            if not os.path.isfile(cache_path):
                cache = {}
                self._logger.warn(f"Cache file '{cache_path}' doesn't exist")
            else:
                # open file
                self._logger.debug(f"Reading cache file '{cache_path}'")
                try:
                    with open(cache_path, 'r') as f:
                        cache = json.load(f)
                except Exception as e:
                    self._logger.warn(f"Failed to open cache file '{cache_path}': {repr(e)}")
                else:
                    speech_path = cache.get(model, {}).get(voice, {}).get(str(speed), {}).get(instructions, {}).get(text)
                    if speech_path is None:
                        self._logger.debug("Speech not found in cache")
                    else:
                        self._logger.debug(f"Found speech in cache '{speech_path}'")

        # generate speech if necessary

        if speech_path is None:
            # validate connection
            if self.parameters.probe_api_connection:
                success, message = self.validate_connection(model=model)
                if not success:
                    return False, message, None

            # retrieve API key
            success, message, api_key = self.retrieve_api_key()
            if not success:
                self._logger.error(message)
                return False, message, None, None

            # use API
            self._logger.debug(f"Retrieving speech from API (text='{text}', model='{model}', voice='{voice}', speed='{speed}', instructions='{instructions}')")
            success, message, speech_bytes = self.speech_post(
                text=text,
                model=model,
                voice=voice,
                speed=speed,
                instructions=instructions,
                api_url=self.api_endpoints[self.parameters.api_endpoint]['speech_url'],
                api_key=api_key
            )
            if not success:
                self._logger.error(message)
                return False, message, None

            # create cache folder

            if not os.path.exists(self.parameters.cache_folder):
                self._logger.debug(f"Creating cache folder '{self.parameters.cache_folder}'")
                try:
                    os.makedirs(self.parameters.cache_folder)
                except Exception as e:
                    self._logger.error(f"Failed to create cache folder '{self.parameters.cache_folder}': {repr(e)}")

            # write speech to file

            stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            speech_path = os.path.join(self.parameters.cache_folder, f"{stamp}.wav")
            self._logger.debug(f"Writing speech to file '{speech_path}'")

            try:
                with open(speech_path, mode='bw') as f:
                    f.write(speech_bytes)
            except Exception as e:
                message = f"Failed to write speech to file '{speech_path}': {repr(e)}"
                self._logger.error(message)
                return False, message, None

            # write path to cache

            if cache_write:
                cache_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)
                self._logger.debug(f"Writing speech path to cache file '{cache_path}'")

                # add path to cache

                if model not in cache:
                    cache[model] = {}
                if voice not in cache[model]:
                    cache[model][voice] = {}
                if str(speed) not in cache[model][voice]:
                    cache[model][voice][str(speed)] = {}
                if instructions not in cache[model][voice][str(speed)]:
                    cache[model][voice][str(speed)][instructions] = {}

                cache[model][voice][str(speed)][instructions][text] = speech_path

                # write cache

                if os.path.exists(self.parameters.cache_folder):
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump(cache, f, indent=4)
                    except Exception as e:
                        self._logger.error(f"Failed to save speech path to cache file '{cache_path}': {repr(e)}")

        # forward results

        self._logger.info(f"Retrieved speech '{speech_path}' (text='{text}', model='{model}', voice='{voice}', speed='{speed}', instructions='{instructions}')")

        return True, "Retrieved speech.", speech_path

    # Callbacks

    def get_speech_callack(self, request, response):
        self._logger.debug("get_speech_callack(): start")

        response.success, response.message, speech_path = self.get_speech(
            text=request.text,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            instructions=request.instructions
        )
        if response.success:
            response.path = speech_path

        self._logger.debug("get_speech_callack(): end")
        return response

def main(args=None):
    start_and_spin_node(Speech, args=args)

if __name__ == '__main__':
    main()
