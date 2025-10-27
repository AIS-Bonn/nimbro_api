#!/usr/bin/env python3

import os
import json
import time

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from nimbro_api_interfaces.srv import TranslationsGet
from nimbro_api.misc.common import validate_default_endpopints, filter_api_endpoint, validate_api_endpoint, retrieve_api_key, probe_models_api, validate_connection

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, log_lines

### <Parameter Defaults>

node_name = "translations"
severity = 10

probe_api_connection = True
api_endpoint = "OpenAI"

## non-params

line_length = 150
log_line_length = 150

api_endpoints = {
    'OpenAI': {
        'api_flavor': "openai",
        'models_url': "https://api.openai.com/v1/models",
        'translations_url': "https://api.openai.com/v1/audio/translations",
        'key_type': "environment",
        'key_value': "OPENAI_API_KEY"
    },
    'vLLM': {
        'api_flavor': "openai",
        'models_url': "http://localhost:8000/v1/models",
        'translations_url': "http://localhost:8000/v1/audio/translations",
        'key_type': "environment",
        'key_value': "VLLM_API_KEY"
    },
    'AIS': {
        'api_flavor': "openai",
        'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
        'translations_url': "https://api-code.ais.uni-bonn.de/v1/audio/translations",
        'key_type': "environment",
        'key_value': "AIS_API_KEY"
    }
}

### </Parameter Defaults>

class Translation(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # initialize endpoints

        self.endpoint_keys_required = {'name', 'api_flavor', 'translations_url', 'key_type', 'key_value'}
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
            name="log_line_length",
            dtype=int,
            default_value=log_line_length,
            description="Maximum line length of selected logger messages.",
            read_only=False,
            range_min=1,
            range_max=999999,
            range_step=1
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
            description=f"Sets the API endpoint defining API flavor, Models & Translation URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}.",
            read_only=False
        )

        # create interfaces

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)

        self.cbg_translations = ReentrantCallbackGroup()
        self.srv_translations = self.create_service(TranslationsGet, f"{self.node_namespace}/{self.node_name}/get_translation".replace("//", "/"), self.get_translation_callback, qos_profile=qos_profile, callback_group=self.cbg_translations)

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

        return value, message

    # Translation Pipeline

    def translations_post(self, path, model, temperature, prompt, response_format, api_url, api_key):
        self._logger.debug(f"Retrieving translation from API (path='{path}', model='{model}', temperature='{temperature}', prompt='{prompt}', response_format='{response_format}')")

        try:
            with open(path, "rb") as f:
                audio_file = f.read()
        except Exception as e:
            message = f"Failed to read file '{path}': {repr(e)}"
            self._logger.error(message)
            return False, message, None

        headers = {
            'Authorization': f"Bearer {api_key}"
        }

        data = {
            'model': model,
            'temperature': temperature,
            'response_format': response_format
        }
        if prompt != "":
            data['prompt'] = prompt

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
        response = requests.post(api_url, headers=headers, files={'file': (os.path.basename(path), audio_file)}, data=data, stream=False)
        toc = time.perf_counter()
        self._logger.debug(f"Received response after '{toc - tic:.3f}s'")

        if response.status_code == 200:
            success = True
            message = "Retrieved translation."
            try:
                translation = json.dumps(response.json(), indent=2)
            except Exception:
                translation = response.text.strip()
        else:
            success = False
            message = f"HTTP-Error '{response.status_code}': {response.text}"
            translation = None

        return success, message, translation

    def get_translation(self, path, model, temperature, prompt, response_format):
        # parse arguments

        if not os.path.isfile(path):
            message = f"Path '{path}' is not a valid file."
            self._logger.error(message)
            return False, message, None

        if self.parameters.api_endpoint == "OpenAI" and model == "":
            model = "whisper-1"
            self._logger.debug(f"Using default OpenAI model '{model}'")

        temperature_range = [0.0, 1.0]
        if temperature < temperature_range[0] or temperature > temperature_range[1]:
            self._logger.warn(f"Clipping temperature '{temperature}' to interval {temperature_range}")
            temperature = max(temperature_range[0], min(temperature_range[1], temperature))

        supported_formats = ["json", "verbose_json", "text", "srt", "vtt"]
        if response_format == "":
            response_format = supported_formats[0]
            self._logger.debug(f"Using default response format '{supported_formats}'")
        elif response_format not in response_format:
            message = f"Response format '{response_format}' is not supported. Supported response formats are: {response_format}"
            self._logger.error(message)
            return False, message, None

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
        success, message, translation = self.translations_post(
            path=path,
            model=model,
            temperature=temperature,
            prompt=prompt,
            response_format=response_format,
            api_url=self.api_endpoints[self.parameters.api_endpoint]['translations_url'],
            api_key=api_key
        )
        if not success:
            log_lines(
                text=message,
                line_length=self.parameters.log_line_length,
                line_highlight="| ",
                block_format=False,
                max_lines=20,
                logger=self._logger,
                severity=40
            )
            return False, message, None

        # forward results

        log_lines(
            text=f"Retrieved translation (path='{path}', model='{model}', temperature='{temperature}', prompt='{prompt}', response_format='{response_format}'):\n{translation}",
            line_length=self.parameters.log_line_length,
            line_highlight="| ",
            block_format=False,
            logger=self._logger,
            severity=20
        )

        return True, "Retrieved translation.", translation

    # Callbacks

    def get_translation_callback(self, request, response):
        self._logger.debug("get_translation_callback(): start")

        # chunking_strategy, known_speaker_names
        response.success, response.message, translation = self.get_translation(
            path=request.path,
            model=request.model,
            temperature=request.temperature,
            prompt=request.prompt,
            response_format=request.response_format
        )
        if response.success:
            response.translation = translation

        self._logger.debug("get_translation_callback(): end")
        return response

def main(args=None):
    start_and_spin_node(Translation, args=args)

if __name__ == '__main__':
    main()
