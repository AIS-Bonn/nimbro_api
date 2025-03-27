#!/usr/bin/env python3

import os
import re
import copy
import json
import time
import base64
import multiprocessing

import requests

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange, FloatingPointRange
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

from nimbro_api_interfaces.msg import ApiUsage
from nimbro_api_interfaces.srv import CompletionsPrompt, CompletionsStop, CompletionsGetTools, CompletionsSetTools, CompletionsGetContext, CompletionsRemoveContext, TriggerFeedback
from nimbro_api.utils.parameter_handler import ParameterHandler
from nimbro_api.utils.node import start_and_spin_node, SelfShutdown
from nimbro_api.utils.string import normalize, extract_json_from_text

### <Parameter Defaults>

node_name = "completions"
logger_level = 10

probe_api_connection = True
api_endpoint = "OpenAI"
model_name = "gpt-4o"

model_temperatur = 1.0
model_top_p = 1.0
model_max_tokens = 1000
model_presence_penalty = 0.0
model_frequency_penalty = 0.0

stream_completion = True
normalize_text_response = True
max_tool_calls_per_response = 1
correction_attempts = 2
timeout_chunk = 15.0 # seconds
timeout_completion = 60.0 # seconds

## non-params

status_interval = 1.0 # seconds
ignore_invalid_function_parameter_enums = False
condense_consecutive_user_messages = True
modify_text_for_json_compliance = True

api_endpoints = {
    'OpenAI': {
        'api_flavor': "openai",
        'models_url': "https://api.openai.com/v1/models",
        'completions_url': "https://api.openai.com/v1/chat/completions",
        'key_type': "environment",
        'key_value': "OPENAI_API_KEY"
    },
    'Mistral AI': {
        'api_flavor': "mistral",
        'models_url': "https://api.mistral.ai/v1/models",
        'completions_url': "https://api.mistral.ai/v1/chat/completions",
        'key_type': "environment",
        'key_value': "MISTRAL_API_KEY"
    },
    'OpenRouter': {
        'api_flavor': "openrouter",
        'models_url': "https://openrouter.ai/api/v1/models",
        'completions_url': "https://openrouter.ai/api/v1/chat/completions",
        'key_type': "environment",
        'key_value': "OPENROUTER_API_KEY"
    },
    'vLLM': {
        'api_flavor': "vllm",
        'models_url': "http://localhost:8000/v1/models",
        'completions_url': "http://localhost:8000/v1/chat/completions",
        'key_type': "environment",
        'key_value': "VLLM_API_KEY"
    }
}

### </Parameter Defaults>

class Completions(Node):

    def __init__(self):
        super().__init__(node_name)
        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self.endpoint_keys_required = {'name', 'api_flavor', 'completions_url', 'key_type', 'key_value'}
        self.endpoint_keys_optional = {'models_url'}
        self.endpoint_key_type_values = ["environment", "plain"]
        self.endpoint_api_flavor_values = ["openai", "mistral", "vllm", "openrouter"]

        assert isinstance(api_endpoints, dict), f"{type(api_endpoints).__name__}"
        endpoint_keys_required = self.endpoint_keys_required - {'name'}
        for endpoint_name in api_endpoints:
            assert isinstance(endpoint_name, str), f"Endpoint names must be of type 'str' instead of '{type(endpoint_name).__name__}'."
            endpoint = api_endpoints[endpoint_name]
            assert isinstance(endpoint, dict), f"Endpoint '{endpoint_name}' must be of type 'dict' instead of '{type(endpoint).__name__}'."
            assert all(isinstance(key, str) for key in endpoint), f"Endpoint '{endpoint_name}' must contain only keys of type 'str' instead of {[type(key).__name__ for key in endpoint]}."
            assert set(endpoint.keys()) >= endpoint_keys_required, f"Endpoint '{endpoint_name}' must contain keys {sorted(endpoint_keys_required)} (and optionally {sorted(self.endpoint_keys_optional)}) instead of {sorted(endpoint.keys())}."
            assert set(endpoint.keys()) <= endpoint_keys_required | self.endpoint_keys_optional, f"Endpoint '{endpoint_name}' must contain keys {sorted(endpoint_keys_required)} (and optionally {sorted(self.endpoint_keys_optional)}) instead of {sorted(endpoint.keys())}."
            assert all(isinstance(endpoint[key], str) for key in endpoint), f"Endpoint '{endpoint_name}' must contain only values of type 'str' instead of {[type(endpoint[key]).__name__ for key in endpoint]}."
            assert endpoint['api_flavor'] in self.endpoint_api_flavor_values, f"Endpoint '{endpoint_name}' must contain key 'api_flavor' with value in {self.endpoint_api_flavor_values} instead of '{endpoint['api_flavor']}'."
            assert endpoint['key_type'] in self.endpoint_key_type_values, f"Endpoint '{endpoint_name}' must contain key 'key_type' with value in {self.endpoint_key_type_values} instead of '{endpoint['key_type']}'."

        self.api_endpoints = api_endpoints
        self.endpoint_probes = {}

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
        descriptor.name = "probe_api_connection"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Probes the Models API of the endpoint to validate the API key and model name."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, probe_api_connection, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "api_endpoint"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = f"Sets the API endpoint defining API flavor, Models & Completions URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, api_endpoint, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "model_name"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Name of the model that is used."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, model_name, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "model_temperatur"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Higher values like will make the output more random, while lower values like will make it more focused and deterministic."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = 0.0
        float_range.to_value = 1.5
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, model_temperatur, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "model_top_p"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "An alternative to sampling with temperature, called nucleus sampling, which behaves similar for similar values."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = 0.0
        float_range.to_value = 2.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, model_top_p, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "model_max_tokens"
        descriptor.type = ParameterType.PARAMETER_INTEGER
        descriptor.description = "Maximum generated tokens per completion. Must be smaller than context size minus input tokens."
        descriptor.read_only = False
        int_range = IntegerRange()
        int_range.from_value = 1
        int_range.to_value = 10000
        int_range.step = 1
        descriptor.integer_range.append(int_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, model_max_tokens, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "model_presence_penalty"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Positive values penalize new tokens based on whether they appear in the text so far."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = -2.0
        float_range.to_value = 2.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, model_presence_penalty, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "model_frequency_penalty"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Positive values penalize new tokens based on their existing frequency in the text so far."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = -2.0
        float_range.to_value = 2.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, model_frequency_penalty, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "stream_completion"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Using streaming to receive completions."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, stream_completion, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "normalize_text_response"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Applies text normalization to text responses (except JSON mode is used) without affecting the internal state of the message history."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, normalize_text_response, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "max_tool_calls_per_response"
        descriptor.type = ParameterType.PARAMETER_INTEGER
        descriptor.description = "A response that is allowed to contain tool calls must contain at most this many tool calls. Set to '0' to deactivate."
        descriptor.read_only = False
        int_range = IntegerRange()
        int_range.from_value = 0
        int_range.to_value = 100
        int_range.step = 1
        descriptor.integer_range.append(int_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, max_tool_calls_per_response, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "correction_attempts"
        descriptor.type = ParameterType.PARAMETER_INTEGER
        descriptor.description = "Invokes self-correction once invalid responses are being generated or timeouts take place."
        descriptor.read_only = False
        int_range = IntegerRange()
        int_range.from_value = 0
        int_range.to_value = 1000
        int_range.step = 1
        descriptor.integer_range.append(int_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, correction_attempts, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "timeout_chunk"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Time in seconds since last received chat completion chunk after which chat completion gets aborted to forward a invalid response."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = 0.5
        float_range.to_value = 1000.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, timeout_chunk, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "timeout_completion"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Time in seconds since chat completion call after which chat completion gets aborted to forward a invalid response."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = 0.5
        float_range.to_value = 1000.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, timeout_completion, descriptor)

        self.logger_level_default = self.logger_level
        self.probe_api_connection_default = self.probe_api_connection
        self.api_endpoints_default = copy.deepcopy(self.api_endpoints)
        self.api_endpoint_default = self.api_endpoint
        self.model_name_default = self.model_name
        self.model_temperatur_default = self.model_temperatur
        self.model_top_p_default = self.model_top_p
        self.model_max_tokens_default = self.model_max_tokens
        self.model_presence_penalty_default = self.model_presence_penalty
        self.model_frequency_penalty_default = self.model_frequency_penalty
        self.stream_completion_default = self.stream_completion
        self.normalize_text_response_default = self.normalize_text_response
        self.max_tool_calls_per_response_default = self.max_tool_calls_per_response
        self.correction_attempts_default = self.correction_attempts
        self.timeout_chunk_default = self.timeout_chunk
        self.timeout_completion_default = self.timeout_completion

        self.parameter_handler.all_declared()

        self.tools = None
        self.messages = []
        self.awaited_tool_responses = []
        self.is_prompting = False

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=50)

        self.cbg_prompt = MutuallyExclusiveCallbackGroup()

        self.srv_prompt = self.create_service(CompletionsPrompt, f"{self.node_namespace}/{self.node_name}/prompt".replace("//", "/"), self.prompt, qos_profile=qos_profile, callback_group=self.cbg_prompt)
        self.srv_stop = self.create_service(CompletionsStop, f"{self.node_namespace}/{self.node_name}/stop".replace("//", "/"), self.stop_response, qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        self.srv_get_tools = self.create_service(CompletionsGetTools, f"{self.node_namespace}/{self.node_name}/get_tools".replace("//", "/"), self.get_tools, qos_profile=qos_profile, callback_group=self.cbg_prompt)
        self.srv_set_tools = self.create_service(CompletionsSetTools, f"{self.node_namespace}/{self.node_name}/set_tools".replace("//", "/"), self.set_tools, qos_profile=qos_profile, callback_group=self.cbg_prompt)

        self.srv_get_context = self.create_service(CompletionsGetContext, f"{self.node_namespace}/{self.node_name}/get_context".replace("//", "/"), self.get_context, qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.srv_remove_context = self.create_service(CompletionsRemoveContext, f"{self.node_namespace}/{self.node_name}/remove_context".replace("//", "/"), self.remove_context, qos_profile=qos_profile, callback_group=self.cbg_prompt)
        self.srv_reset_parameters = self.create_service(TriggerFeedback, f"{self.node_namespace}/{self.node_name}/reset_parameters".replace("//", "/"), self.reset_parameters, qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        qos_profile_pub = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_ALL, depth=10)
        self.pub_usage = self.create_publisher(ApiUsage, f"{self.node_namespace}/api_usage".replace("//", "/"), qos_profile=qos_profile_pub, callback_group=MutuallyExclusiveCallbackGroup())

        self.pub_status = self.create_publisher(DiagnosticStatus, f"{self.node_namespace}/{self.node_name}/status".replace("//", "/"), qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.timer_status = self.create_timer(status_interval, self.publish_status, callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info("Node started")

    def __del__(self):
        self.get_logger().info("Node shutdown")

    def parameter_changed(self, parameter):
        success = True
        reason = ""

        if parameter.name == "logger_level":
            self.logger_level = parameter.value
            rclpy.logging.set_logger_level(f"{self.node_namespace}/{self.node_name}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(self.logger_level))

        elif parameter.name == "probe_api_connection":
            if not self.setup_finished or self.probe_api_connection != parameter.value:
                self.probe_api_connection = parameter.value
                if self.probe_api_connection and self.setup_finished:
                    if self.endpoint_probes.get(self.api_endpoint) is None and self.api_endpoints[self.api_endpoint].get('models_url') is not None:
                        success, reason = self.probe_models_api(self.api_endpoint)
                    else:
                        self.get_logger().debug(f"Probing Models API for endpoint '{self.api_endpoint}' is not required")
                    if success and self.api_endpoint in self.endpoint_probes and self.model_name not in self.endpoint_probes[self.api_endpoint]['models']:
                        self.get_logger().warn(f"Selected model name '{self.model_name}' is not in list of available models: {self.endpoint_probes[self.api_endpoint]['models']}")

        elif parameter.name == "api_endpoint":
            probe = None
            if parameter.value in list(self.api_endpoints.keys()):
                json_object = None
                if self.endpoint_probes.get(parameter.value) is None and self.api_endpoints[parameter.value].get('models_url') is not None:
                    probe = parameter.value
            else:
                success, reason, json_object = self.validate_api_endpoint(parameter.value)
                if success:
                    if json_object['name'] in list(self.api_endpoints.keys()):
                        json_object_without_name = copy.deepcopy(json_object)
                        del json_object_without_name['name']
                        for key in ['models_url', 'key_type', 'key_value']:
                            if self.api_endpoints[json_object['name']][key] != json_object_without_name[key] and json_object.get('models_url') is not None:
                                probe = json_object
                    elif json_object.get('models_url') is not None:
                        probe = json_object

            if success and self.probe_api_connection is True:
                if probe is None:
                    self.get_logger().debug(f"Probing Models API for endpoint '{parameter.value if json_object is None else json_object['name']}' is not required")
                else:
                    success, reason = self.probe_models_api(probe)
                if success and self.setup_finished and self.model_name not in self.endpoint_probes.get(parameter.value if json_object is None else json_object['name'], {'models': [self.model_name]})['models']:
                    self.get_logger().warn(f"Selected model name '{self.model_name}' is not in list of available models: {self.endpoint_probes[parameter.value if json_object is None else json_object['name']]['models']}")

            if success:
                names_before = list(self.api_endpoints.keys())
                dicts_before = copy.deepcopy(self.api_endpoints)

                if json_object is None:
                    self.api_endpoint = parameter.value
                else:
                    self.api_endpoint = json_object['name']
                    self.api_endpoints[self.api_endpoint] = json_object
                    del self.api_endpoints[self.api_endpoint]['name']

                if self.api_endpoint in names_before:
                    if self.api_endpoints[self.api_endpoint] != dicts_before[self.api_endpoint]:
                        self.get_logger().info(f"Updated API endpoint '{self.api_endpoint}'")
                else:
                    self.get_logger().info(f"Created new API endpoint '{self.api_endpoint}'")

        elif parameter.name == "model_name":
            if self.probe_api_connection:
                if parameter.value in self.endpoint_probes.get(self.api_endpoint, {}).get('models', [parameter.value]):
                    self.model_name = parameter.value
                else:
                    success = False
                    reason = f"Model '{parameter.value}' is not in list of available models {self.endpoint_probes[self.api_endpoint]['models']}."

        elif parameter.name == "model_temperatur":
            self.model_temperatur = parameter.value

        elif parameter.name == "model_top_p":
            self.model_top_p = parameter.value

        elif parameter.name == "model_max_tokens":
            self.model_max_tokens = parameter.value

        elif parameter.name == "model_presence_penalty":
            self.model_presence_penalty = parameter.value

        elif parameter.name == "model_frequency_penalty":
            self.model_frequency_penalty = parameter.value

        elif parameter.name == "stream_completion":
            self.stream_completion = parameter.value

        elif parameter.name == "normalize_text_response":
            self.normalize_text_response = parameter.value

        elif parameter.name == "max_tool_calls_per_response":
            self.max_tool_calls_per_response = parameter.value

        elif parameter.name == "correction_attempts":
            self.correction_attempts = parameter.value

        elif parameter.name == "timeout_chunk":
            self.timeout_chunk = parameter.value

        elif parameter.name == "timeout_completion":
            self.timeout_completion = parameter.value

        else:
            return None, None

        return success, reason

    def validate_api_endpoint(self, api_endpoint):
        try:
            json_object = json.loads(api_endpoint)
        except Exception:
            success = False
            message = f"Value must be a name of an existing endpoint in {list(self.api_endpoints.keys())} or a valid JSON encoded dictionary containing a new endpoint."
            json_object = None
        else:
            if not isinstance(json_object, dict):
                success = False
                message = f"JSON encoded endpoint must be of type 'dict' instead of '{type(json_object).__name__}'."
            elif not all(isinstance(key, str) for key in json_object):
                success = False
                message = f"JSON encoded endpoint must contain only values of type 'str' instead of {[type(key).__name__ for key in json_object]}."
            elif not set(json_object.keys()) >= self.endpoint_keys_required:
                success = False
                message = f"JSON encoded endpoint must contain keys {sorted(self.endpoint_keys_required)} (and optionally {sorted(self.endpoint_keys_optional)}) instead of {sorted(json_object.keys())}."
            elif not set(json_object.keys()) <= self.endpoint_keys_required | self.endpoint_keys_optional:
                success = False
                message = f"JSON encoded endpoint must contain keys {sorted(self.endpoint_keys_required)} (and optionally {sorted(self.endpoint_keys_optional)}) instead of {sorted(json_object.keys())}."
            elif not all(isinstance(json_object[key], str) for key in json_object):
                success = False
                message = f"JSON encoded endpoint must contain only values of type 'str' instead of {[type(json_object[key]).__name__ for key in json_object]}."
            elif not json_object['api_flavor'] in self.endpoint_api_flavor_values:
                success = False
                message = f"JSON encoded endpoint must contain key 'api_flavor' with value in {self.endpoint_api_flavor_values} instead of '{json_object['api_flavor']}'."
            elif not json_object['key_type'] in self.endpoint_key_type_values:
                success = False
                message = f"JSON encoded endpoint must contain key 'key_type' with value in {self.endpoint_key_type_values} instead of '{json_object['key_type']}'."
            else:
                success = True
                message = ""

        return success, message, json_object

    def probe_models_api(self, api_endpoint):
        if isinstance(api_endpoint, str):
            api_endpoint_name = api_endpoint
            api_endpoint = self.api_endpoints[api_endpoint]
        else:
            api_endpoint_name = api_endpoint['name']

        success = True
        message = ""

        if api_endpoint['key_type'] == "environment":
            api_key = os.getenv(api_endpoint['key_value'])
            if api_key is None:
                success = False
                message = f"Error while probing Models API: Failed to read API key from environment variable '{api_endpoint['key_value']}')."
        else:
            api_key = api_endpoint['key_value']

        if success:
            self.get_logger().debug(f"Probing Models API '{api_endpoint['models_url']}' of endpoint '{api_endpoint_name}' using key '{api_key}'")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            try:
                response = requests.get(api_endpoint['models_url'], headers=headers)
            except Exception as e:
                success = False
                message = f"Error while probing Models API: Failed to retrieve available models: {e}"
            else:
                if response.status_code != 200:
                    success = False
                    message = f"Error while probing Models API: Received unexpected HTTP status code '{response.status_code}' with message '{response.text}'."
                else:
                    # self.get_logger().debug(f"{json.dumps(response.json(), indent=4)}")
                    available_models = [m['id'] for m in response.json()['data']]
                    if len(available_models) == 0:
                        self.get_logger().warn("There are no models available")
                    else:
                        self.get_logger().debug(f"Available models: {available_models}")
                    self.endpoint_probes[api_endpoint_name] = {'models': available_models, 'stamp': time.time()}

        if not success:
            self.get_logger().error(message)
            if api_endpoint_name in self.endpoint_probes:
                del self.endpoint_probes[api_endpoint_name]

        return success, message

    # Prompt Pipeline

    def update_awaited_tool_responses(self):
        all_ids = []
        awaited_tool_responses = []
        for i, message in enumerate(self.messages):
            if message['role'] == 'tool':
                assert 'tool_call_id' in message, f"{message}"
                if message['tool_call_id'] in awaited_tool_responses:
                    awaited_tool_responses.remove(message['tool_call_id'])
                else:
                    self.get_logger().warn("The message history contains a tool response '{message}' without a corresponding tool call")
            elif message['role'] == 'assistant':
                if 'tool_calls' in message:
                    for call in message['tool_calls']:
                        assert 'id' in call, f"{call}"
                        all_ids.append(call['id'])
                        awaited_tool_responses.append(call['id'])

        num_duplicates = len(all_ids) - len(set(all_ids))
        if num_duplicates > 0:
            self.get_logger().warn(f"The message history contains '{num_duplicates}' tool call{'s' if num_duplicates != 1 else ''} with an ID of another tool call")

        self.awaited_tool_responses = awaited_tool_responses
        self.get_logger().debug(f"Awaiting tool response{'' if len(self.awaited_tool_responses) == 1 else 's'}: {None if len(self.awaited_tool_responses) == 0 else self.awaited_tool_responses}")

    def check_prompt_validity(self, request):
        response = CompletionsPrompt.Response()
        response.success = False
        response.message = ""
        response.text = ""
        response.tool_calls = []

        text = request.text.strip()
        self.get_logger().debug(f"Received request (role='{request.role}', text='{text}', reset_context='{request.reset_context}', tool_response_id='{request.tool_response_id}', response_type='{request.response_type}')")

        if request.role == 'json':
            try:
                message = json.loads(text)
            except Exception as e:
                response.message = f"Invalid request - Field 'role' is set to 'json' but 'text' field cannot be parsed as JSON ({e})."
                self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                return None, response
            else:
                try:
                    self.check_message_validity(message)
                except Exception as e:
                    response.message = f"Invalid request - {e}"
                    self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                    return None, response
                try:
                    message = self.encode_local_images(message)
                except Exception as e:
                    response.message = f"Invalid request - {e}"
                    self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                    return None, response
        else:
            message = None

        # role is unknown
        if request.role not in ['system', 'user', 'assistant', 'tool', 'json']:
            response.message = f"Invalid request - Unknown role '{request.role}'. Valid roles are 'system', 'user', 'assistant', 'tool', and 'json'."
            self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
            return None, response

        if request.role != "json":

            # cannot respond tool call and reset conversation
            if request.tool_response_id != "" and request.reset_context:
                response.message = f"Invalid request - 'reset_context' cannot be 'True' while 'tool_response_id' is not-empty string '{request.tool_response_id}'."
                self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                return None, response

            # cannot respond to tool call that is not awaited
            if (request.tool_response_id != "" and request.tool_response_id not in self.awaited_tool_responses) and not request.reset_context:
                response.message = f"Invalid request - Not awaiting tool response with ID '{request.tool_response_id}'. Awaiting tool response IDs: '{self.awaited_tool_responses}."
                self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                return None, response

            # enforce response to awaited tool call
            if len(self.awaited_tool_responses) > 0 and not request.reset_context:
                if request.tool_response_id not in self.awaited_tool_responses: # TODO is order important if len(self.awaited_tool_responses) > 1?
                    response.message = f"Invalid request - Awaiting tool response IDs '{self.awaited_tool_responses}'."
                    self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                    return None, response

            # tool_response_id is not awaited
            if request.tool_response_id != "" and request.tool_response_id not in self.awaited_tool_responses:
                response.message = f"Invalid request - Unknown tool response ID '{request.tool_response_id}'. Awaiting tool response IDs '{self.awaited_tool_responses}'."
                self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                return None, response

            # tool_response_id must be empty if role is not tool
            if request.tool_response_id != "" and request.role != "tool":
                response.message = "Invalid request - Tool responses must use role 'tool', not '" + request.role + "'."
                self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                return None, response

        # enforce response_type 'none', 'text', 'auto' if no tools are defined
        if self.tools is None and request.response_type != "none" and request.response_type != "text" and request.response_type != "json" and request.response_type != "auto":
            response.message = "Invalid request - No tools are defined, so field 'response_type' must be set to 'none', 'text', or 'auto'."
            self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
            return None, response

        # enforce response type 'none', 'text', 'auto', 'always', or valid tool name if tools are defined
        if (self.tools is not None) and request.response_type != "none" and request.response_type != "text" and request.response_type != "json" and request.response_type != "auto" and request.response_type != "always":
            found = False
            for f in self.tools:
                if f['function']['name'] == request.response_type:
                    found = True
                    break
            if not found:
                response.message = "Invalid request - 'response_type' field must be 'none', 'text', 'auto', 'always', or a valid tool name'."
                self.get_logger().error(f"Failed to prompt model ({response.message[:-1]})")
                return None, response

        self.get_logger().debug("Received request is valid")
        return message, None

    def check_message_validity(self, message):
        if not isinstance(message, dict):
            raise Exception(f"Message must be of type 'dict' but it is of type {type(message).__name__}.")
        if 'role' not in message:
            raise Exception("Message must contain key 'role'.")
        if message['role'] not in ['system', 'user', 'assistant', 'tool']:
            raise Exception(f"Message must contain key 'role' with value in ['system', 'user', 'assistant', 'tool'] but it is '{message['role']}'.")

        if message['role'] == 'system':
            if 'content' not in message:
                raise Exception("System message must contain key 'content'.")
            if not isinstance(message['content'], str):
                raise Exception(f"System message value of key 'content' must be of type 'str' but it is of type {type(message['content']).__name__}.")
            if len(message['content']) == 0:
                raise Exception("System message value of key 'content' must not be empty.")

            if 'name' in message:
                if not isinstance(message['name'], str):
                    raise Exception(f"System message can contain key 'name' with value that must be of type 'str' but it is of type {type(message['name']).__name__}.")
                if len(message['name']) == 0:
                    raise Exception("System message can contain key 'name' with a value that must not be empty.")

            for key in message:
                if key not in ['role', 'content', 'name']:
                    raise Exception(f"System message keys must be in ['role', 'content', 'name'] which '{key}' is not.")

        if message['role'] == 'user':
            if 'content' not in message:
                raise Exception("User message must contain key 'content'.")
            if not isinstance(message['content'], str) and not isinstance(message['content'], list):
                raise Exception(f"User message value of key 'content' must be of type 'str' or 'list' but it is of type {type(message['content']).__name__}.")

            if isinstance(message['content'], list):
                for element in message['content']:
                    if not isinstance(element, dict):
                        raise Exception(f"User message content elements must be of type 'dict' but it is of type {type(element).__name__}.")
                    if 'type' not in element:
                        raise Exception("User message content element must contain key 'type'.")
                    if element['type'] not in ["text", "image_url"]:
                        raise Exception(f"User message content element type must be in ['text', 'image_url'] but it is '{element['type']}'.")
                    if element['type'] == "text":
                        if 'text' not in element:
                            raise Exception("User message content element of type text must contain key 'text'.")
                        if not isinstance(element['text'], str):
                            raise Exception(f"User message content element of type text must contain key 'text' of type 'str' but it is of type '{type(element['text']).__name__}'.")
                        if len(element['text']) == 0:
                            raise Exception("User message content element of type text must contain key 'text' that is not empty.")
                        if not len(element) == 2:
                            raise Exception(f"User message content element of type text must contain exactly two keys 'type' and 'text' but it contains {list(element.keys())}.")
                    else:
                        if 'image_url' not in element:
                            raise Exception("User message content element of type image_url must contain key 'image_url'.")
                        if not isinstance(element['image_url'], dict):
                            raise Exception(f"User message content element of type image_url must be of type 'dict' but it is of type {type(element['image_url']).__name__}.")
                        if self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm"]:
                            if 'detail' in element['image_url']:
                                del element['image_url']['detail'] # TODO this function should not do anything
                                # raise Exception("User message content element of type image_url must not contain key 'detail'.")
                        else:
                            if 'detail' not in element['image_url']:
                                raise Exception("User message content element of type image_url must contain key 'detail'.")
                            if not element['image_url']['detail'] in ["low", "high", "auto"]:
                                raise Exception(f"User message content element of type image_url must contain key 'detail' with value in ['low', 'high', 'auto'] but it is '{element['image_url']['detail']}'.")
                        if 'url' not in element['image_url']:
                            raise Exception("User message content element of type image_url must contain key 'url'.")
                        if not isinstance(element['image_url']['url'], str):
                            raise Exception(f"User message content element of type image_url must contain key 'url' of type 'str' but it is of type '{type(element['image_url']['url']).__name__}'.")
                        if len(element['image_url']['url']) == 0:
                            raise Exception("User message content element of type image_url must contain key 'url' that is not empty.")
                        if self.api_endpoints[self.api_endpoint]['api_flavor'] == "vllm":
                            if not len(element['image_url']) == 1:
                                raise Exception(f"User message content element of type image_url must contain exactly one key 'url' but it contains {list(element['image_url'].keys())}.")
                        else:
                            if not len(element['image_url']) == 2:
                                raise Exception(f"User message content element of type image_url must contain exactly two keys 'detail' and 'url' but it contains {list(element['image_url'].keys())}.")

            if 'name' in message:
                if not isinstance(message['name'], str):
                    raise Exception(f"User message can contain key 'name' with value that must be of type 'str' but it is of type {type(message['name']).__name__}.")
                if len(message['name']) == 0:
                    raise Exception("User message can contain key 'name' with a value that must not be empty.")

            for key in message:
                if key not in ['role', 'content', 'name']:
                    raise Exception(f"User message keys must be in ['role', 'content', 'name'] which '{key}' is not.")

        if message['role'] == 'assistant':
            if 'content' not in message:
                raise Exception("Assistant message must contain key 'content'.")
            if message['content'] is None:
                if 'tool_calls' not in message:
                    raise Exception("Assistant message can only contain key 'content' with value 'None' if it also contains key 'tool_calls'.")
            elif isinstance(message['content'], str):
                if len(message['content']) == 0:
                    raise Exception("Assistant message value of key 'content' must not be an empty string.")
            else:
                raise Exception(f"Assistant message must contain key 'content' with value of type 'None' or 'str' but it is of type '{type(message['content']).__name__}'.")

            if 'tool_calls' in message:
                if not isinstance(message['tool_calls'], list):
                    raise Exception(f"Assistant message key 'tool_calls' must be of type 'list' but it is of type '{type(message['tool_calls']).__name__}'.")
                for element in message['tool_calls']:
                    if not isinstance(element, dict):
                        raise Exception(f"Assistant message elements of key 'tool_calls' must be of type 'dict' but it is of type '{type(element).__name__}'.")
                    if 'id' not in element:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain key 'id'.")
                    if not isinstance(element['id'], str):
                        raise Exception(f"Assistant message elements of key 'tool_calls' must contain key 'id' with value of type 'str' but it of type '{type(element['id']).__name__}'.")
                    if 'type' not in element:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain key 'type'.")
                    if element['type'] != "function":
                        raise Exception("Assistant message elements of key 'tool_calls' must contain key 'type' with value 'function' but it is '{element['type']}'.")
                    if 'function' not in element:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain key 'function'.")
                    if not isinstance(element['function'], dict):
                        raise Exception(f"Assistant message elements of key 'tool_calls' must contain key 'function' with value of type 'dict' but it of type '{type(element['function']).__name__}'.")
                    if 'name' not in element['function']:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'name'.")
                    if not isinstance(element['function']['name'], str):
                        raise Exception(f"Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'name' with value of type 'str' but it of type '{type(element['function']['name']).__name__}'.")
                    if len(element['function']['name']) == 0:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'name' with value that must not be empty.")
                    if 'arguments' not in element['function']:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'arguments'.")
                    if not isinstance(element['function']['arguments'], str):
                        raise Exception(f"Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'arguments' with value of type 'str' but it of type '{type(element['function']['arguments']).__name__}'.")
                    if len(element['function']['arguments']) == 0:
                        raise Exception("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'arguments' with value that must not be empty.")
                    if not len(element['function']) == 2:
                        raise Exception(f"Assistant message elements of key 'tool_calls' must contain dict 'function' with exactly two keys 'name' and 'arguments' but it contains {list(element['function'].keys())}.")
                    if not len(element) == 3:
                        raise Exception(f"Assistant message elements of key 'tool_calls' must contain exactly three keys 'id', 'type' and 'functions' but it contains {list(element.keys())}.")

            if 'name' in message:
                if not isinstance(message['name'], str):
                    raise Exception(f"Assistant message can contain key 'name' with value that must be of type 'str' but it is of type {type(message['name']).__name__}.")
                if len(message['name']) == 0:
                    raise Exception("Assistant message can contain key 'name' with a value that must not be empty.")

            for key in message:
                if key not in ['role', 'content', 'tool_calls', 'name']:
                    raise Exception(f"Assistant message keys must be in ['role', 'content', 'tool_calls', 'name'] which '{key}' is not.")

        if message['role'] == 'tool':
            if 'content' not in message:
                raise Exception("Tool message must contain key 'content'.")
            if not isinstance(message['content'], str):
                raise Exception(f"Tool message value of key 'content' must be of type 'str' but it is of type {type(message['content']).__name__}.")
            if len(message['content']) == 0:
                raise Exception("Tool message value of key 'content' must not be empty.")
            if 'tool_call_id' not in message:
                raise Exception("Tool message must contain key 'tool_call_id'.")
            if not isinstance(message['tool_call_id'], str):
                raise Exception(f"Tool message value of key 'tool_call_id' must be of type 'str' but it is of type {type(message['tool_call_id']).__name__}.")
            if len(message['tool_call_id']) == 0:
                raise Exception("Tool message value of key 'tool_call_id' must not be empty.")
            if message['tool_call_id'] not in self.awaited_tool_responses:
                raise Exception("Tool message value of key 'tool_call_id' must be in list of awaited responses {self.awaited_tool_responses} but it is '{message['tool_call_id']}'.")
            if not len(message) == 3:
                raise Exception(f"Tool message must contain exactly three keys 'role', 'content' and 'tool_call_id' but it contains {list(message.keys())}.")

    def encode_local_images(self, message):
        if message['role'] == "user":
            if isinstance(message['content'], list):
                for i, element in enumerate(message['content']):
                    if element['type'] == "image_url":
                        if os.path.isfile(element['image_url']['url']):
                            self.get_logger().debug(f"Provided image URL '{element['image_url']['url']}' points to local file")
                            try:
                                with open(element['image_url']['url'], "rb") as image_file:
                                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            except Exception as e:
                                raise Exception(f"Failed to encode image URL '{element['image_url']['url']}' ({e}).")
                            else:
                                message['content'][i]['image_url']['url'] = f"data:image/jpeg;base64,{base64_image}"
                        else:
                            self.get_logger().warn(f"Provided image URL '{element['image_url']['url']}' points to web")

        return message

    def add_request_to_context(self, request, message=None):
        if request.reset_context:
            self.messages = []
            self.update_awaited_tool_responses()
            self.get_logger().debug("Reset conversation")

        self.message_length_original = len(self.messages)

        if request.role == "json":
            new_message = message
        else:
            text = request.text.strip()

            if request.role == "system":
                new_message = {"role": request.role, "content": text} # optional name

            elif request.role == "user":
                new_message = {"role": request.role, "content": [{"type": "text", "text": text}]} # optional name
                # new_message = {"role": request.role, "content": text} # optional name

            elif request.role == "assistant":
                new_message = {"role": request.role, "content": text}

            elif request.role == "tool":
                new_message = {"role": request.role, "tool_call_id": request.tool_response_id, "content": text}
            else:
                raise Exception(f"Encountered unexpected role '{request.role}'")

        try:
            self.check_message_validity(new_message)
        except Exception as e:
            self.get_logger().warn(f"Unexpected error in validity check of request message '{new_message}' ({e})")

        self.messages.append(new_message)

        # log

        print_message = copy.deepcopy(new_message)
        if print_message['role'] == "user":
            if isinstance(print_message['content'], list):
                for i, element in enumerate(print_message['content']):
                    if element['type'] == "image_url":
                        print_message['content'][i]['image_url']['url'] = "<IMAGE>"
        print_message = str(print_message).replace("\n", "\\n")

        self.get_logger().info(f"Request added to message history: '{print_message}'")

    def get_no_completion_response(self):
        response = CompletionsPrompt.Response()
        response.success = True
        response.message = "The service request's content was added to the message history without generating a response."
        response.text = ""
        response.tool_calls = []
        # self.get_logger().info(response.message[:-1])
        return response

    def generate_completion(self, request):
        self.set_tool_choice(request)

        is_valid = True
        corrections = 0

        while True:
            if not is_valid:
                corrections += 1
                self.get_logger().warn(f"Starting correction attempt '{corrections}' of '{self.correction_attempts}'")

            text, tool_calls = "", []
            is_complete = False

            self.pipe = multiprocessing.Pipe()
            completion_proc = multiprocessing.Process(target=self.completion_process)
            completion_proc.daemon = True

            start_prompt = time.perf_counter()
            last_chunk = time.perf_counter()
            completion_proc.start()

            while True:
                now = time.perf_counter()

                if now - start_prompt > self.timeout_completion:
                    self.pipe[0].send("INTERNAL")
                    message = f"Error while receiving response (Timeout after '{self.timeout_completion}s' before response was completed)."
                    break

                if self.stream_completion is True and now - last_chunk > self.timeout_chunk:
                    self.pipe[0].send("INTERNAL")
                    message = f"Error while receiving response (Timeout after '{self.timeout_chunk}s' without receiving a chunk)."
                    break

                if self.pipe[0].poll():
                    last_chunk = time.perf_counter()

                    chunk = self.pipe[0].recv()
                    assert isinstance(chunk, dict), f"Expected chunk '{chunk}' to be of type 'dict' instead of '{type(chunk).__name__}'"
                    assert set(chunk.keys()) == {'code', 'content'}, f"Expected chunk '{chunk}' to have keys 'code' and 'content'"
                    assert isinstance(chunk['code'], str), f"Expected chunk code '{chunk['code']}' to be of type 'str' instead of '{type(chunk['code']).__name__}'"
                    assert chunk['code'] in ['ERROR', 'STOP', 'USAGE', 'COMPLETION', 'ALL_CHUNKS_RECEIVED'], f"Expected chunk code '{chunk['code']}' to be in ['ERROR', 'STOP', 'COMPLETION', 'ALL_CHUNKS_RECEIVED']"
                    # assert isinstance(chunk['content'], str), f"Expected chunk content '{chunk['content']}' to be of type 'str' instead of '{type(chunk['content']).__name__}'"

                    if chunk['code'] == "USAGE":
                        print()
                        if 'prompt_tokens_details' in chunk['content'] and chunk['content']['prompt_tokens_details'] is None:
                            del chunk['content']['prompt_tokens_details']
                        self.save_usage(
                            prompt_tokens=chunk['content']['prompt_tokens'],
                            prompt_tokens_cached=chunk['content'].get('prompt_tokens_details', {}).get('cached_tokens', 0),
                            completion_tokens=chunk['content']['completion_tokens']
                        )

                    if chunk['code'] == "ERROR" or chunk['code'] == 'STOP':
                        if chunk['code'] == 'STOP':
                            response = self.post_process_completion(request, None, None, False, corrections if chunk['code'] == "ERROR" else -1, chunk['content'])
                            return None, None, None, None, None, response
                        message = chunk['content']
                        break

                    elif chunk['code'] == "COMPLETION":
                        text, tool_calls = self.parse_completion_chunk(chunk['content'], text, tool_calls)

                    elif chunk['code'] == "ALL_CHUNKS_RECEIVED":
                        is_complete = True

                        # extract tool calls from text
                        if len(tool_calls) == 0 and (request.response_type == "always" or (request.response_type != "text" and request.response_type != "auto" and request.response_type != "json")):
                            text, tool_calls = self.extract_tool_call_from_text(text, tool_calls)
                        tool_calls = self.clean_tool_call_names(tool_calls)

                        # extract JSON from text
                        if request.response_type == "json" and modify_text_for_json_compliance:
                            try:
                                json.loads(text)
                            except Exception:
                                dict_extracted = extract_json_from_text(text)
                                if dict_extracted is not None:
                                    self.get_logger().warn(f"Modifying response to conform to JSON. Original response:\n\n{text}\n")
                                    text = json.dumps(dict_extracted, indent=4)

                        self.add_response_to_context(text, tool_calls)
                        break
                else:
                    time.sleep(0.1)

            if is_complete:
                is_valid, _message, correction_response = self.check_response_validity(request, text, tool_calls)
                if not is_valid:
                    message = _message
                if (not is_valid) and corrections < self.correction_attempts:
                    self.get_logger().debug(f"Adding correction message{'' if len(correction_response) == 1 else 's'} to message hisotry:\n" + str('\n'.join([str(element) for element in correction_response])))
                    self.messages += correction_response
                else:
                    try:
                        message
                    except UnboundLocalError:
                        message = _message
                    break
            else:
                self.get_logger().error(message[:-1])
                is_valid = False
                if corrections >= self.correction_attempts:
                    # text, tool_calls = "", []
                    break

        return text, tool_calls, is_valid, corrections, message, None

    def set_tool_choice(self, request):
        if self.api_endpoints[self.api_endpoint]['api_flavor'] == "openai":
            if request.response_type == "text":
                self.response_format = {"type": "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {"type": "json_object"}
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {"type": "text"}
                self.tool_choice = "required"
            elif request.response_type == "auto":
                self.response_format = {"type": "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {"type": "text"}
                self.tool_choice = {"type": "function", "function": {"name": request.response_type}}

        elif self.api_endpoints[self.api_endpoint]['api_flavor'] == "mistral":
            if request.response_type == "text":
                self.response_format = {"type": "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {"type": "json_object"}
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {"type": "text"}
                # self.tool_choice = "required"
                self.tool_choice = "any"
            elif request.response_type == "auto":
                self.response_format = {"type": "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {"type": "text"}
                self.tool_choice = {"type": "function", "function": {"name": request.response_type}}

        elif self.api_endpoints[self.api_endpoint]['api_flavor'] == "openrouter":
            if request.response_type == "text":
                self.response_format = {"type": "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {"type": "json_object"}
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {"type": "text"}
                self.tool_choice = "required"
            elif request.response_type == "auto":
                self.response_format = {"type": "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {"type": "text"}
                self.tool_choice = {"type": "function", "function": {"name": request.response_type}}

        elif self.api_endpoints[self.api_endpoint]['api_flavor'] == "vllm":
            if request.response_type == "text":
                self.response_format = {"type": "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                # self.response_format = {"type": "json_object"}
                self.response_format = {"type": "text"} # Because the actual JSON-mode in VLLM sucks AFAIK. This still triggers correction for JSON decode errors and deactivates normalization.
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {"type": "text"}
                self.tool_choice = "auto"
                self.get_logger().warn(f"Tool choice '{request.response_type}' is not available for api_flavor '{self.api_endpoints[self.api_endpoint]['api_flavor']}', using '{self.tool_choice}' instead")
            elif request.response_type == "auto":
                self.response_format = {"type": "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {"type": "text"}
                # self.tool_choice = {"type": "function", "function": {"name": request.response_type}}
                request.response_type = "always"
                self.tool_choice = "auto"
                self.get_logger().warn(f"Tool choice '{request.response_type}' is not available for api_flavor '{self.api_endpoints[self.api_endpoint]['api_flavor']}', using '{self.tool_choice}' instead")

        else:
            self.get_logger().fatal(f"Undefined API flavor '{self.api_endpoints[self.api_endpoint]['api_flavor']}'")
            raise SelfShutdown

    def completion_process(self):
        self.get_logger().debug("completion_process(): start")

        if self.api_endpoints[self.api_endpoint]['key_type'] == "environment":
            api_key = os.getenv(self.api_endpoints[self.api_endpoint]['key_value'])
            if api_key is None:
                message = f"Error while sending prompt (Failed to read API key from environment variable '{self.api_endpoints[self.api_endpoint]['key_value']}')."
                self.pipe[1].send({'code': "ERROR", 'content': message})
                return
        else:
            api_key = self.api_endpoints[self.api_endpoint]['key_value']
        self.get_logger().debug(f"Retrieved API key '{api_key}'")

        messages = self.messages

        if condense_consecutive_user_messages:
            while True:
                is_user = 0
                for i in range(len(messages)):
                    if messages[i]['role'] == "user":
                        is_user += 1
                    if is_user > 1 and (messages[i]['role'] != "user" or i == len(messages) - 1):
                        first = i - is_user
                        last = i - 1
                        if i == len(messages) - 1:
                            first += 1
                            last += 1
                        contents = []
                        for j in range(first, last + 1, 1):
                            for k in range(len(messages[j]['content'])):
                                contents.append(messages[j]['content'][k])
                        self.get_logger().debug(f"Condensing '{len(contents)}' consecutive user messages ('{first}' to '{last}') into a single one")
                        new_message = {"role": "user", "content": contents}
                        messages = messages[: first] + [new_message] + messages[last + 1:]
                        break
                    elif messages[i]['role'] != "user":
                        is_user = 0
                else:
                    break

        messages_print = copy.deepcopy(messages)
        for i, message in enumerate(messages_print):
            if isinstance(message['content'], list):
                for j, element in enumerate(message['content']):
                    if element['type'] == "image_url":
                        messages_print[i]['content'][j]['image_url']['url'] = "<IMAGE>"
            messages_print[i] = str(messages_print[i]).replace("\n", "\\n")

        self.get_logger().debug("Sending messages:\n" + str('\n'.join(messages_print)))
        self.get_logger().info(f"Prompting LLM with message {messages_print[-1]} (Messages='{len(messages_print)}')")

        stream = copy.copy(self.stream_completion)

        try:
            headers = {
                'Content-Type': "application/json",
                'Authorization': f"Bearer {api_key}"
            }

            if self.api_endpoints[self.api_endpoint]['api_flavor'] == "openai":
                data = {
                    'model': self.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'temperature': self.model_temperatur,
                    'top_p': self.model_top_p,
                    'max_completion_tokens': self.model_max_tokens,
                    'presence_penalty': self.model_presence_penalty,
                    'frequency_penalty': self.model_frequency_penalty,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': stream
                }
                if self.tools is not None:
                    data['tool_choice'] = self.tool_choice
                    data['parallel_tool_calls'] = False
                if stream is True:
                    data['stream_options'] = {'include_usage': True}

            elif self.api_endpoints[self.api_endpoint]['api_flavor'] == "mistral":
                data = {
                    'model': self.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'tool_choice': self.tool_choice,
                    'temperature': self.model_temperatur,
                    'top_p': self.model_top_p,
                    'max_tokens': self.model_max_tokens,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': stream
                }

            elif self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                data = {
                    'model': self.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'temperature': self.model_temperatur,
                    'top_p': self.model_top_p,
                    'max_tokens': self.model_max_tokens,
                    'presence_penalty': self.model_presence_penalty,
                    'frequency_penalty': self.model_frequency_penalty,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': stream
                }
                if self.tools is not None:
                    data['tool_choice'] = self.tool_choice
                if stream is True:
                    data['stream_options'] = {'include_usage': True}
            else:
                message = f"Error while sending prompt (Undefined API flavor '{self.api_endpoints[self.api_endpoint]['api_flavor']}')."
                self.pipe[1].send({'code': "ERROR", 'content': message})
                return

            self.get_logger().debug("Sending POST request")
            completion = requests.post(self.api_endpoints[self.api_endpoint]['completions_url'], headers=headers, json=data, stream=stream)

            if not stream:
                self.get_logger().debug("Received POST response")
                if completion.status_code != 200:
                    message = f"Received unexpected HTTP status code '{completion.status_code}' with message '{completion.text}'"
                    self.pipe[1].send({'code': "ERROR", 'content': message})
                else:
                    try:
                        json_data = completion.json()
                    except Exception as e:
                        message = f"Failed to parse response as JSON: {e}"
                        self.pipe[1].send({'code': "ERROR", 'content': message})
                    else:
                        if 'choices' not in json_data:
                            message = "Expected response to contain key 'choices'"
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif not isinstance(json_data['choices'], list):
                            message = f"Expected response value of key 'choices' to be of type 'list' instead of '{type(json_data['choices']).__name__}'"
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif len(json_data['choices']) == 0:
                            message = "Expected response list 'choices' to contain at least one element"
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif 'message' not in json_data['choices'][0]:
                            message = "Expected response choice to contain key 'message'"
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif not isinstance(json_data['choices'][0]['message'], dict):
                            message = f"Expected response choice value of key 'message' to be of type 'dict' instead of '{type(json_data['choices'][0]['message']).__name__}'"
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        else:
                            self.get_logger().debug("POST response:\n" + json.dumps(json_data, indent=4))
                            completion = json_data['choices'][0]['message']
                            if 'tool_calls' in completion and completion['tool_calls'] is None:
                                del completion['tool_calls']
                            self.pipe[1].send({'code': "COMPLETION", 'content': completion})
                            if 'usage' in json_data:
                                self.pipe[1].send({'code': "USAGE", 'content': json_data['usage']})
                            else:
                                self.get_logger().warn("Reponse did not contain usage")
                            self.pipe[1].send({'code': "ALL_CHUNKS_RECEIVED", 'content': ''})

                self.get_logger().debug("completion_process(): end")
                return

        except Exception as e:
            message = f"Error while sending prompt (Failed to post request: {e})."
            self.pipe[1].send({'code': "ERROR", 'content': message})

        else:
            self.get_logger().debug("Sent POST request")

            decoded_buffer = ""
            undecoded_buffer = b""
            error = ""
            early_stop = False
            usage = None

            for chunk in completion.iter_content(chunk_size=1):
                # self.get_logger().debug(f"chunk: {chunk}")
                if early_stop is True:
                    break
                decoded = False

                # check if response was canceled from external source
                if self.pipe[1].poll():
                    code = self.pipe[1].recv()
                    if code == "EXTERNAL":
                        message = "Response was stopped due to request from external source."
                        self.get_logger().debug(message[:-1])
                        self.pipe[1].send({'code': "STOP", 'content': message})
                    else:
                        self.get_logger().debug("Response was stopped due to request from internal source")
                    break

                # attempt to decode chunk
                if chunk:
                    if len(undecoded_buffer) > 0:
                        try:
                            decoded_chunk = (undecoded_buffer + chunk).decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                decoded_chunk = chunk.decode('utf-8')
                            except UnicodeDecodeError:
                                undecoded_buffer += chunk
                            else:
                                decoded = True
                                self.get_logger().warn(f"Ignoring byte sequence '{undecoded_buffer}' after failure to decode it")
                                undecoded_buffer = b""
                        else:
                            decoded = True
                            undecoded_buffer = b""
                    else:
                        try:
                            decoded_chunk = chunk.decode('utf-8')
                        except UnicodeDecodeError:
                            undecoded_buffer += chunk
                        else:
                            decoded = True

                # process all decoded lines
                if decoded:
                    decoded_buffer += decoded_chunk
                    # self.get_logger().debug(f"{"\n" in decoded_buffer}: decoded_buffer: {decoded_buffer.replace("\n", "\\n")}")
                    while '\n' in decoded_buffer:
                        line, decoded_buffer = decoded_buffer.split('\n', 1)
                        # self.get_logger().debug(f"line: {line}")
                        if line != "":
                            if line.find('data:') == 0:

                                # end of response
                                if line == 'data: [DONE]':
                                    # forward usage before end of process
                                    if self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                        if usage is None:
                                            self.get_logger().warn("Did not receive usage before [DONE] message")
                                        else:
                                            self.pipe[1].send({'code': "USAGE", 'content': usage})
                                    self.get_logger().debug("Received [DONE] message")
                                    self.pipe[1].send({'code': "ALL_CHUNKS_RECEIVED", 'content': ''})
                                else:
                                    try:
                                        json_data = json.loads(line[6:])
                                    except Exception as e:
                                        self.get_logger().warn(f"Ignoring line '{line}' after failure to parse it as JSON ({e})")
                                    else:
                                        # unexpected finish reason
                                        if json_data.get('finish_reason') not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                                            # forward usage before end of process
                                            if self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                                if usage is None:
                                                    self.get_logger().warn("Did not receive usage before [ERROR] message")
                                                else:
                                                    self.pipe[1].send({'code': "USAGE", 'content': usage})
                                            message = f"Error while receiving response: Unexpected finish reason '{json_data.get('finish_reason')}'."
                                            self.pipe[1].send({'code': "ERROR", 'content': message})
                                            early_stop = True
                                            break
                                        else:
                                            # extract usage
                                            if json_data.get('usage') is not None:
                                                if self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                                    usage = json_data['usage']
                                                else:
                                                    self.pipe[1].send({'code': "USAGE", 'content': json_data['usage']})

                                            # extract choices
                                            if len(json_data.get('choices', [])) > 0:
                                                try:
                                                    json_choice = json_data["choices"][0]
                                                except Exception as e:
                                                    self.get_logger().warn(f"Ignoring data '{json_data}' after failure to parse choice as JSON ({e})")
                                                else:
                                                    # unexpected finish reason
                                                    if json_choice.get('finish_reason') not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                                                        # forward usage before end of process
                                                        if self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                                            if usage is None:
                                                                self.get_logger().warn("Did not receive usage before [ERROR] message")
                                                            else:
                                                                self.pipe[1].send({'code': "USAGE", 'content': usage})
                                                        message = f"Error while receiving response: Unexpected finish reason '{json_choice.get('finish_reason')}'."
                                                        self.pipe[1].send({'code': "ERROR", 'content': message})
                                                        early_stop = True
                                                        break
                                                    else:
                                                        # forward delta
                                                        self.pipe[1].send({'code': "COMPLETION", 'content': json_choice["delta"]})
                            else:
                                error += line
            else:
                self.get_logger().debug("Received full POST response")

                if len(undecoded_buffer) > 0:
                    self.get_logger().warn(f"Ignoring byte sequence '{undecoded_buffer}' after failure to decode it")

                if len(decoded_buffer) > 0:
                    error += decoded_buffer

                # forward remaining usage before end of process
                if usage is not None and self.api_endpoints[self.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                    self.pipe[1].send({'code': "USAGE", 'content': usage})

                # forward collected error
                if error != "":
                    message = f"Error while receiving response ({error})."
                    self.pipe[1].send({'code': "ERROR", 'content': message})

            completion.close()
            self.get_logger().debug("Connection closed")

        self.get_logger().debug("completion_process(): end")

    def save_usage(self, prompt_tokens, prompt_tokens_cached, completion_tokens):
        self.get_logger().info(f"Completion consumed '{prompt_tokens}' prompt- and '{completion_tokens}' completion tokens")

        usage = ApiUsage()
        usage.api_type = "completions"
        usage.api_endpoint = self.api_endpoint
        usage.model_name = self.model_name
        usage.input_tokens_uncached = prompt_tokens - prompt_tokens_cached
        usage.input_tokens_cached = prompt_tokens_cached
        usage.output_tokens = completion_tokens

        self.pub_usage.publish(usage)

    def parse_completion_chunk(self, chunk, text, tool_calls):
        # self.get_logger().debug(f"\nchunk:\n\n{chunk}\n")

        # response is text
        if 'content' in chunk and chunk['content'] is not None:
            # self.get_logger().debug("Received content chunk: '" + str(chunk['content']).replace("\n", "\\n") + "'")
            print(chunk['content'], end="", flush=True)
            if not isinstance(chunk['content'], str):
                raise Exception(f"Expected tool_calls to be of type 'list' instead of '{type(chunk['tool_calls']).__name__}': {chunk}")
            text += chunk['content']

        # response is tool call
        if 'tool_calls' in chunk:
            # self.get_logger().debug("Received tool chunk: '" + str(chunk['tool_calls']).replace("\n", "\\n") + "'")
            if not isinstance(chunk['tool_calls'], list):
                raise Exception(f"Expected tool_calls to be of type 'list' instead of '{type(chunk['tool_calls']).__name__}': {chunk}")
            # if len(chunk['tool_calls']) == 0:
            #     raise Exception(f"Expected tool_calls to be a list of length '1' instead of '{len(chunk['tool_calls'])}': {chunk}")
            for i in range(len(chunk['tool_calls'])):
                if 'index' in chunk['tool_calls'][i] and 'function' in chunk['tool_calls'][i]:
                    if 'id' in chunk['tool_calls'][i] and 'name' in chunk['tool_calls'][i]['function']:
                        if len(tool_calls) == chunk['tool_calls'][i]['index']:
                            tool_calls.append({"id": chunk['tool_calls'][i]['id'], "name": chunk['tool_calls'][i]['function']['name'], 'arguments': ""})
                        else:
                            raise Exception(f"Expected tool_calls elements field 'index' to have the value '{len(tool_calls)}' instead of '{chunk['tool_calls'][i]['index']}'")
                    if 'arguments' in chunk['tool_calls'][i]['function']:
                        if chunk['tool_calls'][i]['index'] < len(tool_calls):
                            tool_calls[chunk['tool_calls'][i]['index']]['arguments'] += chunk['tool_calls'][i]['function']['arguments']
                        else:
                            raise Exception(f"Expected tool_calls elements field 'index' to be smaller '{len(tool_calls)}' instead of '{chunk['tool_calls'][i]['index']}'")
                elif set(chunk['tool_calls'][i].keys()) == {'id', 'type', 'function'}:
                    tool_calls.append({"id": chunk['tool_calls'][i]['id'], "name": chunk['tool_calls'][i]['function']['name'], "arguments": chunk['tool_calls'][i]['function']['arguments']})
                else:
                    raise Exception(f"Expected tool_calls element to contain the the fields 'index' and 'function', or 'id', 'type' and 'function' instead of {list(chunk['tool_calls'][i].keys())}")

        return text, tool_calls

    def extract_tool_call_from_text(self, text, tool_calls):
        first_text_call = extract_json_from_text(text, first_over_longest=True)
        if first_text_call is not None:
            if 'name' in first_text_call and 'arguments' in first_text_call:
                self.get_logger().warn("Detected tool call in text output while expecting tool call and not having received one, moving it to tool calls and erasing text")
                try:
                    text = text.replace(first_text_call, '')
                except Exception:
                    text = ''
                if 'id' in first_text_call:
                    first_text_call['id'] = first_text_call['id']
                else:
                    self.get_logger().warn("Tool call misses 'id' field, automatically generating an ID")
                    made_up_id = self.get_clock().now().seconds_nanoseconds()
                    made_up_id = f"{made_up_id[0]}_{made_up_id[1]}"
                    first_text_call['id'] = made_up_id
                first_text_call['arguments'] = json.dumps(first_text_call['arguments'])
                tool_calls.append(first_text_call)

        return text, tool_calls

    def clean_tool_call_names(self, tool_calls):
        # I experienced openai referring to undefined functions names in a way that includes special characters (e.g. 'assistant.tell_joke' instead of 'tell_joke').
        # Responding to such a function would cause the completion to respond with 'invalid function name' due to the illegal use of special characters.
        # So, we remove special characters here, establish a legal function name, and then let the self correction routines check validity w.r.t. the defined JSON scheme.
        for i, call in enumerate(tool_calls):
            if not re.match('^[a-zA-Z0-9_-]{1,64}$', call["name"]):
                tool_calls[i]["name"] = re.sub(r"[^a-zA-Z0-9_-]", "", call["name"])
                tool_calls[i]["name"] = tool_calls[i]["name"][:64]
                self.get_logger().warn("Encountered invalid function name '" + call["name"] + "', renaming it to '" + tool_calls[i]["name"] + "'")

        return tool_calls

    def add_response_to_context(self, text, tool_calls):
        if text != "":
            print()
            self.get_logger().info(f"Response:\nText:\n\n{text}\n\nTools: '{tool_calls}'")
        else:
            self.get_logger().info(f"Response:\nText: ''\nTools: '{tool_calls}'")

        message = {}
        message['role'] = "assistant"
        if text == "":
            message['content'] = None
        else:
            message['content'] = text
        if len(tool_calls) > 0:
            message['tool_calls'] = [{} for _ in range(len(tool_calls))]
        for i in range(len(tool_calls)):
            message['tool_calls'][i]['type'] = "function"
            message['tool_calls'][i]['id'] = tool_calls[i]['id']
            message['tool_calls'][i]['function'] = {}
            message['tool_calls'][i]['function']['name'] = tool_calls[i]['name']
            message['tool_calls'][i]['function']['arguments'] = tool_calls[i]['arguments']

        try:
            self.check_message_validity(message)
        except Exception as e:
            self.get_logger().warn(f"Unexpected error in validity check of response message '{message}' ({e})")

        self.messages.append(message)
        self.get_logger().debug(("Response added to message history: '" + str(message)).replace("\n", "\\n").strip() + "'")

    def check_response_validity(self, request, text, tool_calls):
        is_valid = True
        response_message = ""

        # create generic correction response # TODO have a single correction response rather than number-of-tools-calls + 1 single messages

        correction_response = []
        tool_call_is_valid_default_correction = "This tool call is valid and does not require any correction."
        for i, call in enumerate(tool_calls):
            correction_response.append({})
            correction_response[-1]["role"] = "tool"
            correction_response[-1]["tool_call_id"] = call['id']
            correction_response[-1]["content"] = tool_call_is_valid_default_correction
        correction_response.append({})
        correction_response[-1]["role"] = "user"
        correction_response[-1]["content"] = "Your response is invalid. Please correct it based on the provided error messages and try again!"

        # test error cases

        # error case: tool use when there should not be any tool use
        if (self.tools is None or request.response_type == "text") and len(tool_calls) > 0:
            is_valid = False
            response_message = "Response contains a tool call despite " + ("no tools being defined" if self.tools is None else "tool choice was set to 'none'") + "."
            self.get_logger().error(response_message[:-1])
            for i in range(len(correction_response)):
                if 'tool_call_id' in correction_response[i]:
                    correction_response[i]["content"] = "Your response must not contain any tool call, but only text content."

        # error case: tool choice "use specific function" was violated
        if self.tools is not None and request.response_type != "text" and request.response_type != "auto" and request.response_type != "always" and request.response_type != "json":
            if text != "":
                is_valid = False
                response_message = "Response contains text content despite tool choice being set to '" + request.response_type + "'."
                self.get_logger().error(response_message[:-1])
                correction_response[-1]["content"] = "Your response must only contain a tool call of '" + request.response_type + "' without additional text."
            else:
                valid_ids = []
                invalid_ids_names = {}
                for c in tool_calls:
                    if c["name"] == request.response_type:
                        valid_ids.append(c["id"])
                    else:
                        invalid_ids_names[c["id"]] = c["name"]
                for i in range(len(correction_response)):
                    if correction_response[i]["role"] == "tool":
                        if not correction_response[i]['tool_call_id'] in valid_ids:
                            is_valid = False
                            response_message = "Response contains tool call '" + invalid_ids_names[correction_response[i]['tool_call_id']] + "' despite tool choice being set to '" + request.response_type + "'."
                            self.get_logger().error(response_message[:-1])
                            correction_response[i]["content"] = "Your response must only contain the tool call '" + request.response_type + "'."

        # error case: exceeding maximum number of tool calls per response
        if len(tool_calls) > self.max_tool_calls_per_response and self.max_tool_calls_per_response > 0:
            is_valid = False
            response_message = "Response contains '" + str(len(tool_calls)) + "' tool calls, but the maximum number of tool calls per response is '" + str(self.max_tool_calls_per_response) + "'."
            self.get_logger().error(response_message[:-1])
            for i in range(len(correction_response)):
                if 'tool_call_id' in correction_response[i]:
                    correction_response[i]["content"] = "A valid response must contain at most " + str(self.max_tool_calls_per_response) + " tool call" + ("" if self.max_tool_calls_per_response == 1 else "s") + ", but yours contains " + str(len(tool_calls)) + " tool calls. Please filter accordingly and try again!"

        # error case: custom tool choice "always" was violated
        if request.response_type == "always" and len(tool_calls) == 0:
            is_valid = False
            response_message = "Response does not contain a tool call despite tool choice being set to value 'always'."
            self.get_logger().error(response_message[:-1])
            for i in range(len(correction_response)):
                if 'tool_call_id' not in correction_response[i]:
                    if self.max_tool_calls_per_response == 1:
                        correction_response[i]["content"] = "Please express your last message in a tool call instead of a text response!"
                    else:
                        correction_response[i]["content"] = "Your response must contain " + ("at least one" if self.max_tool_calls_per_response > 1 else "a") + " tool call. Please try again!"

        # error case: function call violates JSON scheme
        for i, call in enumerate(tool_calls):
            for j in range(len(correction_response)):
                if 'tool_call_id' in correction_response[j]:
                    if call["id"] == correction_response[j]['tool_call_id']:
                        if correction_response[j]["content"] == tool_call_is_valid_default_correction:
                            valid, reason, parameters = self.check_tool_call_validity([call["name"], call["arguments"]])
                            if not valid:
                                is_valid = False
                                correction_response[j]["content"] = reason
                                response_message = reason
                        else:
                            self.get_logger().debug("Skipping JSON-scheme based validity check of tool call '" + call["name"] + "' as it is already considered invalid by some previous filter")

        # error case: text response cannot be parsed as JSON despite JSON-mode being activated
        if request.response_type == "json":
            try:
                json.loads(text)
            except Exception as e:
                is_valid = False
                response_message = f"Response cannot be parsed as JSON despite response type being set to JSON ({e})."
                self.get_logger().error(response_message[:-1])
                correction_response[-1]["content"] = "Your response is invalid because it cannot be parsed as JSON. Please try again and respond only with valid JSON and no additional text."
            else:
                self.get_logger().debug("Response parses as JSON")

        return is_valid, response_message, correction_response

    def check_tool_call_validity(self, tool_call):
        valid = True
        message = ""
        parameters = {}

        if not isinstance(tool_call, list):
            valid = False
            message = "A function call should be of type 'list'."
        elif not len(tool_call) == 2:
            valid = False
            message = "A function call should be a list of length '2'."
        else:
            found = None
            if self.tools is not None:
                for i in range(len(self.tools)): # TODO lock to not update functions between calling model and evaluating response
                    if self.tools[i]["function"]["name"] == tool_call[0]:
                        found = i
                        break
            if found is None:
                valid = False
                message = f"'{tool_call[0]}' is not a valid function name."
            else:
                try:
                    parameters = json.loads(tool_call[1])
                except Exception as e:
                    valid = False
                    message = f"Failed to parse arguments as JSON ({e})."
                else:
                    for p in parameters.keys():

                        if p not in self.tools[i]["function"]["parameters"]["properties"].keys():
                            valid = False
                            message = f"'{p}' is not a valid argument to this function."
                            break

                        if isinstance(parameters[p], str):
                            if not self.tools[i]["function"]["parameters"]["properties"][p]["type"] == "string":
                                valid = False
                                message = f"Argument '{p}' should be of type 'string' but is of type '{type(parameters[p]).__name__}'."
                                break
                        elif isinstance(parameters[p], bool):
                            if not self.tools[i]["function"]["parameters"]["properties"][p]["type"] == "boolean":
                                valid = False
                                message = f"Argument '{p}' should be of type 'boolean' but is of type '{type(parameters[p]).__name__}'."
                                break
                        elif isinstance(parameters[p], (int, float)):
                            if not self.tools[i]["function"]["parameters"]["properties"][p]["type"] == "number":
                                valid = False
                                message = f"Argument '{p}' should be of type 'number' but is of type '{type(parameters[p]).__name__}'."
                                break
                        else:
                            valid = False
                            message = f"Argument '{p}' is of unsupported type '{type(parameters[p]).__name__}'."
                            break

                        if "enum" in self.tools[i]["function"]["parameters"]["properties"][p].keys():
                            if not parameters[p] in self.tools[i]["function"]["parameters"]["properties"][p]["enum"]:
                                if ignore_invalid_function_parameter_enums:
                                    self.get_logger().warn(f"Ignoring invalid value '{parameters[p]}' of argument '{p}' in function '{tool_call[0]}'")
                                else:
                                    valid = False
                                    message = f"Value '{parameters[p]}' of argument '{p}' is not valid. " + ((f"Values for argument '{p}' must be in list {self.tools[i]['function']['parameters']['properties'][p]['enum']}.\nAlternatively, you might want to call a different function.") if len(self.tools[i]['function']['parameters']['properties'][p]['enum']) else f"Currently, there exists no valid value for argument '{p}', please call a function other then '{tool_call[0]}' instead.")
                                    break

                    if "required" in self.tools[i]["function"]['parameters']:
                        for p in self.tools[i]["function"]['parameters']['required']:
                            if p not in parameters.keys():
                                valid = False
                                message = f"This function requires an argument '{p}'."
                                break

        if valid:
            self.get_logger().debug(f"Function call '{tool_call}' is valid")
        else:
            self.get_logger().error(f"Function call '{tool_call}' is not valid ({message[:-1]})")

        return valid, message, parameters

    def post_process_completion(self, request, text, tool_calls, is_valid, corrections, message):
        if not is_valid:
            self.messages = self.messages[:self.message_length_original]
            text = ""
            tool_calls = []
            if corrections == -1:
                self.get_logger().info("Forwarding empty response after completion was intentionally stopped")
            else:
                self.get_logger().info(f"Forwarding empty response after '{corrections}' correction attempt{'' if corrections == 1 else 's'} without finding a valid response")

            if len(self.messages) != self.message_length_original:
                self.get_logger().error(f"Expected message history to contain '{self.message_length_original}' message{'s' if self.message_length_original == 1 else 's'} but it contains '{len(self.messages)}'")
        elif corrections > 0:
            message = f"Response valid after '{corrections}' self correction{'' if corrections == 1 else 's'}. Last error before correction: {message}"
            self.messages = self.messages[:self.message_length_original + 1] + [self.messages[-1]]
            if len(self.messages) != self.message_length_original + 2:
                self.get_logger().error(f"Expected message history to contain '{self.message_length_original + 2}' messages but it contains '{len(self.messages)}'")
        else:
            if len(self.messages) != self.message_length_original + 2:
                self.get_logger().error(f"Expected message history to contain '{self.message_length_original + 2}' messages but it contains '{len(self.messages)}'")

        if is_valid:
            self.get_logger().debug(f"Forwarding valid response after '{corrections}' correction attempt{'' if corrections == 1 else 's'}")

        self.update_awaited_tool_responses()

        response = CompletionsPrompt.Response()
        response.success = is_valid
        response.message = message
        response.text = text
        for i, call in enumerate(tool_calls):
            tool_calls[i] = json.dumps(call)
        response.tool_calls = tool_calls

        if self.normalize_text_response and request.response_type != "json":
            response.text = normalize(response.text)
        if response.success and response.message == "":
            response.message = "Prompt was successful."

        return response

    # Callbacks

    def prompt(self, request, response):
        self.get_logger().debug("prompt(): start")

        assert not self.is_prompting
        self.is_prompting = True

        self.update_awaited_tool_responses()

        message, check_prompt_response = self.check_prompt_validity(request)
        if check_prompt_response is not None:
            response = check_prompt_response
            self.is_prompting = False
            self.get_logger().debug("prompt(): end")
            return response

        self.add_request_to_context(request, message)

        if request.response_type == "none":
            response = self.get_no_completion_response()
            self.is_prompting = False
            self.get_logger().debug("prompt(): end")
            return response

        text, tool_calls, is_valid, corrections, message, completion_response = self.generate_completion(request)
        if completion_response is not None:
            response = completion_response
            self.is_prompting = False
            self.get_logger().debug("prompt(): end")
            return response

        response = self.post_process_completion(request, text, tool_calls, is_valid, corrections, message)

        self.is_prompting = False
        self.get_logger().debug("prompt(): end")
        return response

    def stop_response(self, request, response):
        self.get_logger().debug("stop_response(): start")

        if self.is_prompting:
            while True:
                try:
                    self.pipe[0].send("EXTERNAL")
                except Exception:
                    pass

                if self.is_prompting:
                    self.get_logger().debug("Waiting until running prompt is stopped", throttle_duration_sec=1.0, skip_first=True)
                    time.sleep(0.1)
                else:
                    break

            response.success = True
            response.message = "The running prompt was stopped."
        else:
            response.success = True
            response.message = "There is no prompt in progress that can be stopped."
            self.get_logger().info("Ignored attempt to stop prompt because no prompt is running")

        self.get_logger().debug("stop_response(): end")
        return response

    def get_tools(self, request, response):
        self.get_logger().info("Received request to forward tools")

        if self.tools is None:
            response.success = True
            response.message = "There are no tools defined."
            response.tools = []
        else:
            try:
                response.tools = [json.dumps(tool['function']) for tool in self.tools]
            except Exception as e:
                response.success = False
                response.message = f"Failed to parse tools as JSON ({e})."
                response.tools = []
            else:
                response.success = True
                response.message = "Successfully retrieved tools."

        return response

    def set_tools(self, request, response):
        response.success = True
        response.message = "Successfully updated the defined tools."

        self.get_logger().debug("Received tool update")

        if len(request.tools) == 0:
            response.message = "Successfully deactivated all tools."
            if self.tools is None:
                self.get_logger().info("Ignored attempt to deactivate tools which they are already")
            else:
                self.tools = None
                self.get_logger().info("Tools deactivated")

        else:
            tools = []

            for i in range(len(request.tools)):
                try:
                    tools.append(json.loads(request.tools[i]))
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to parse function '{request.tools[i]}' as JSON ({e})."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                if set(tools[-1].keys()) != {'parameters', 'name', 'description'}:
                    response.success = False
                    response.message = f"Function '{i}' does not satisfy the required format - The top level keys must be 'parameters', 'name', and 'description' and not '{tools[-1].keys()}'."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                if not isinstance(tools[-1]['name'], str):
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'name' must be of type 'str' and not '{type(tools[-1]['name']).__name__}'."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                if not isinstance(tools[-1]['description'], str):
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'description' must be of type 'str' and not '{type(tools[-1]['description']).__name__}'."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                if not isinstance(tools[-1]['parameters'], dict):
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'parameters' must be of type 'dict' and not '{type(tools[-1]['parameters']).__name__}'."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                if set(tools[-1]['parameters']) != {'type', 'properties', 'required'} and set(tools[-1]['parameters']) != {'type', 'properties'}:
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'parameters' must contain the keys 'type', 'properties', and optionally 'required'."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                if tools[-1]['parameters']['type'] != "object":
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'parameters'::'type' must have the value 'object'."
                    self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                    break

                for p in tools[-1]['parameters']['properties'].keys():

                    if not isinstance(tools[-1]['parameters']['properties'][p], dict):
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'properties'::'{p}' must be of type 'dict' and not '{type(tools[-1]['parameters']['properties'][p]).__name__}'."
                        self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                        break

                    if set(tools[-1]['parameters']['properties'][p].keys()) != {'type', 'description'} and set(tools[-1]['parameters']['properties'][p].keys()) != {'type', 'description', 'enum'}:
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'properties'::'{p}' must contain the keys 'type', 'description', and optionally 'enum'."
                        self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                        break

                    if not isinstance(tools[-1]['parameters']['properties'][p]['description'], str):
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'properties'::'{p}'::'description' must be of type 'str' and not '{type(tools[-1]['parameters']['properties'][p]['description']).__name__}'."
                        self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                        break

                    if not tools[-1]['parameters']['properties'][p]['type'] in ['boolean', 'string', 'number']:
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'properties'::'{p}'::'type' must be of type 'boolean','string', or 'number'."
                        self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                        break

                    if 'enum' in tools[-1]['parameters']['properties'][p].keys():

                        if not isinstance(tools[-1]['parameters']['properties'][p]['enum'], list):
                            response.success = False
                            response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'properties'::'{p}'::'enum' must by of type 'list'."
                            self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                            break

                        if tools[-1]['parameters']['properties'][p]['type'] == 'string':
                            t = str
                        elif tools[-1]['parameters']['properties'][p]['type'] == 'boolean':
                            t = bool
                        else:
                            t = (int, float)

                        for e in tools[-1]['parameters']['properties'][p]['enum']:
                            if not isinstance(e, t):
                                response.success = False
                                response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'properties'::'{p}'::'enum' must only contain elements of type '{t}' and not '{type(e).__name__}'."
                                self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                                break

                if 'required' in tools[-1]['parameters'].keys():
                    if not isinstance(tools[-1]['parameters']['required'], list):
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - The field 'required' must be of type 'list'."
                        self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                        break

                    for r in tools[-1]['parameters']['required']:

                        if not isinstance(r, str):
                            response.success = False
                            response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - All elements in list 'required' must by of type 'str' and not '{type(r).__name__}'."
                            self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                            break

                        if r not in tools[-1]['parameters']['properties'].keys():
                            response.success = False
                            response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format - All elements in list 'required' must refer to an element in 'properties', unlike '{r}'."
                            self.get_logger().error(f"Failed to set tools ({response.message[:-1]})")
                            break

            if response.success is True:
                for i, f in enumerate(tools):
                    tools[i] = {"type": "function", "function": f}

                if self.tools is None:
                    self.get_logger().info("Activated tools:\n" + str('\n'.join([str(tool) for tool in tools])))
                else:
                    self.get_logger().info("Updated tools:\n" + str('\n'.join([str(tool) for tool in tools])))
                self.tools = tools

        return response

    def get_context(self, request, response):
        self.get_logger().info("Received request to forward message history")

        response.success = True
        response.message = "Retrieved message history."
        messages = copy.deepcopy(self.messages)
        response.context = [json.dumps(m) for m in messages]
        response.messages = len(self.messages)

        return response

    def remove_context(self, request, response):
        if request.remove_all:
            response.success = True
            if len(self.messages) == 0:
                response.message = "The message history is already empty."
                self.get_logger().info("Ignored attempt to clear message history that is already empty")
            else:
                self.messages = []
                self.update_awaited_tool_responses()
                response.message = "The message history was cleared."
                self.get_logger().info(response.message[:-1])
            return response

        if request.indexing_last_to_first:
            i = len(self.messages) - request.index - 1
        else:
            i = request.index

        if i < 0 or i > len(self.messages) - 1:
            response.success = False
            response.message = f"Cannot remove message '{i}' from message history because index is out of bounds for message history containing '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."
            self.get_logger().error(f"Failed to remove messages ({response.message[:-1]})")
        else:
            if self.messages[i]['role'] == 'tool':
                self.get_logger().warn("Removing message from message history that is a tool response")
                self.update_awaited_tool_responses()
            elif self.messages[i]['role'] == 'assistant' and 'tool_calls' in self.messages[i]:
                self.get_logger().warn("Removing message from message history that contains a tool call")
                self.update_awaited_tool_responses()

            response.success = True
            message_print = self.messages.pop(i)
            if isinstance(message_print['content'], list):
                for j, element in enumerate(message_print['content']):
                    if element['type'] == "image_url":
                        message_print['content'][j]['image_url']['url'] = "<IMAGE>"
                message_print = str(message_print).replace("\n", "\\n")
            response.message = f"Removed message '{i}' ('{message_print}') from message history."
            self.get_logger().info(response.message[:-1])

        return response

    def reset_parameters(self, request, response):
        self.get_logger().debug("Resetting all parameters to default values")

        response.success = True
        response.message = ""

        keep_probes = []
        for endpoint_name in self.api_endpoints:
            if self.api_endpoints_default.get(endpoint_name) == self.api_endpoints[endpoint_name]:
                keep_probes.append(endpoint_name)

        self.api_endpoints = copy.deepcopy(self.api_endpoints_default)
        for probe_name in copy.deepcopy(self.endpoint_probes):
            if probe_name not in keep_probes:
                del self.endpoint_probes[probe_name]

        params = [
            Parameter('logger_level', Parameter.Type.INTEGER, self.logger_level_default),
            Parameter('probe_api_connection', Parameter.Type.BOOL, self.probe_api_connection_default),
            Parameter('api_endpoint', Parameter.Type.STRING, self.api_endpoint_default),
            Parameter('model_name', Parameter.Type.STRING, self.model_name_default),
            Parameter('model_temperatur', Parameter.Type.DOUBLE, self.model_temperatur_default),
            Parameter('model_top_p', Parameter.Type.DOUBLE, self.model_top_p_default),
            Parameter('model_max_tokens', Parameter.Type.INTEGER, self.model_max_tokens_default),
            Parameter('model_presence_penalty', Parameter.Type.DOUBLE, self.model_presence_penalty_default),
            Parameter('model_frequency_penalty', Parameter.Type.DOUBLE, self.model_frequency_penalty_default),
            Parameter('stream_completion', Parameter.Type.BOOL, self.stream_completion_default),
            Parameter('normalize_text_response', Parameter.Type.BOOL, self.normalize_text_response_default),
            Parameter('max_tool_calls_per_response', Parameter.Type.INTEGER, self.max_tool_calls_per_response_default),
            Parameter('correction_attempts', Parameter.Type.INTEGER, self.correction_attempts_default),
            Parameter('timeout_chunk', Parameter.Type.DOUBLE, self.timeout_chunk_default),
            Parameter('timeout_completion', Parameter.Type.DOUBLE, self.timeout_completion_default)
        ]

        results = self.set_parameters(params)
        for i in range(len(results)):
            if not results[i].successful:
                response.success = False
                log = f"Cannot set parameter '{params[i].name}' to value '{params[i].value}' ({results[i].reason})."
                self.get_logger().error(log[:-1])
                response.message += "\n" + log

        if response.success:
            response.message = "Reset all parameters to default values."
            self.get_logger().info(response.message[:-1])
        else:
            response.message = response.message.strip()
            self.get_logger().warn("Resetting all parameters to default values incomplete")

        return response

    # Session Status

    def publish_status(self):
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK # OK, WARN, ERROR, STALE
        status.name = self.node_name
        status.message = "status"
        status.hardware_id = "llm"

        kv = KeyValue()
        kv.key = "Stamp (seconds.nanoseconds)"
        now = self.get_clock().now().seconds_nanoseconds()
        kv.value = f"{now[0]}.{now[1]}"
        status.values.append(kv)

        kv = KeyValue()
        kv.key = "is_prompting"
        kv.value = f"{self.is_prompting}"
        status.values.append(kv)

        if self.tools is None:
            function_count = 0
        else:
            function_count = len(self.tools)

        for i in range(function_count):
            kv = KeyValue()
            kv.key = "function " + str(i)
            kv.value = str(self.tools[i])
            status.values.append(kv)

        kv = KeyValue()
        kv.key = "function count"
        kv.value = str(function_count)
        status.values.append(kv)

        try:
            self.messages
        except Exception:
            messages = []
        else:
            messages = self.messages

        message_count = len(messages)

        for i in range(message_count):
            kv = KeyValue()
            kv.key = "message " + str(i)
            kv.value = str(messages[i])
            status.values.append(kv)

        kv = KeyValue()
        kv.key = "message count"
        kv.value = str(len(messages))
        status.values.append(kv)

        kv = KeyValue()
        kv.key = "awaiting tool responses"
        kv.value = str(self.awaited_tool_responses)
        status.values.append(kv)

        self.pub_status.publish(status)

def main(args=None):
    start_and_spin_node(Completions, args=args)

if __name__ == '__main__':
    main()
