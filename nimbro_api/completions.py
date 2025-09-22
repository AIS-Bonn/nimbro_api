#!/usr/bin/env python3

import os
import re
import copy
import json
import time
import datetime
import subprocess
import multiprocessing

import requests
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_prefix
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from std_msgs.msg import String

from nimbro_api_interfaces.srv import CompletionsPrompt, CompletionsInterrupt, CompletionsToolsGet, CompletionsToolsSet, CompletionsContextGet, CompletionsContextSet, TriggerFeedback
from nimbro_api.misc.common import validate_default_endpopints, filter_api_endpoint, validate_api_endpoint, retrieve_api_key, probe_models_api, validate_connection, CustomException

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, normalize_string, remove_whitespace, is_base64, is_url, extract_json, read_as_b64, convert_stamp, log_lines, escape

### <Parameter Defaults>

node_name = "completions"

severity = 10
log_line_length = 150
log_last_messages = 0
log_chunks = False

probe_api_connection = True
api_endpoint = "OpenRouter"
model_name = "google/gemini-2.5-flash"

model_temperature = 1.0
model_top_p = 1.0
model_max_tokens = 5000
model_presence_penalty = 0.0
model_frequency_penalty = 0.0
model_reasoning_effort = "none"

completion_parsers = [""]
completion_parsers_timeout = 5.0 # seconds
completion_parsers_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "nimbro_api", "misc", "parsers", "completion")

stream_completion = True
normalize_text_completion = False
max_tool_calls_per_completion = 1
correction_attempts = 0
timeout_chunk_first = 10.0 # seconds
timeout_chunk_next = 5.0 # seconds
timeout_completion = 20.0 # seconds

## non-params

status_interval = 1.0 # seconds

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
    },
    'AIS': {
        'api_flavor': "vllm",
        'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
        'completions_url': "https://api-code.ais.uni-bonn.de/v1/chat/completions",
        'key_type': "environment",
        'key_value': "AIS_API_KEY"
    }
}

### </Parameter Defaults>

class Completions(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # initialize endpoints

        self.endpoint_keys_required = {'name', 'api_flavor', 'completions_url', 'key_type', 'key_value'}
        self.endpoint_keys_optional = {'models_url'}
        self.endpoint_key_type_values = ["environment", "plain"]
        self.endpoint_api_flavor_values = ["openai", "mistral", "vllm", "openrouter"]
        validate_default_endpopints.__get__(self)(api_endpoints)

        self.filter_api_endpoint = filter_api_endpoint.__get__(self)
        self.validate_api_endpoint = validate_api_endpoint.__get__(self)
        self.retrieve_api_key = retrieve_api_key.__get__(self)
        self.probe_models_api = probe_models_api.__get__(self)
        self.validate_connection = validate_connection.__get__(self)

        self.api_endpoints = api_endpoints
        self.endpoint_probes = {}

        # declare parameters

        self.parameter_handler = ParameterHandler(self, settings={'severity': 20, 'log_init_as_debug': True})

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
            name="log_last_messages",
            dtype=int,
            default_value=log_last_messages,
            description="Number of newest messages in context logged with CompletionsPrompt request. Set -1 to log entire context.",
            read_only=False,
            range_min=-1,
            range_max=999999,
            range_step=1
        )

        self.parameter_handler.declare(
            name="log_chunks",
            dtype=bool,
            default_value=log_chunks,
            description="Log all received chunks as DEBUG message.",
            read_only=False
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
            description=f"Sets the API endpoint defining API flavor, Models & Completions URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="model_name",
            dtype=str,
            default_value=model_name,
            description="Name of the model that is used.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="model_temperature",
            dtype=float,
            default_value=model_temperature,
            description="Higher values like will make the output more random, while lower values like will make it more focused and deterministic.",
            read_only=False,
            range_min=0.0,
            range_max=1.5,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="model_top_p",
            dtype=float,
            default_value=model_top_p,
            description="An alternative to sampling with temperature, called nucleus sampling, which behaves similar for similar values.",
            read_only=False,
            range_min=0.0,
            range_max=2.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="model_max_tokens",
            dtype=int,
            default_value=model_max_tokens,
            description="Maximum number of tokens allowed to be generated for one Chat Completion.",
            read_only=False,
            range_min=1,
            range_max=999999999,
            range_step=1
        )

        self.parameter_handler.declare(
            name="model_presence_penalty",
            dtype=float,
            default_value=model_presence_penalty,
            description="Positive values penalize new tokens based on whether they appear in the text so far.",
            read_only=False,
            range_min=-2.0,
            range_max=2.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="model_frequency_penalty",
            dtype=float,
            default_value=model_frequency_penalty,
            description="Positive values penalize new tokens based on their existing frequency in the text so far.",
            read_only=False,
            range_min=-2.0,
            range_max=2.0,
            range_step=0.0
        )

        valid_values = ["", "none", "low", "medium", "high"]
        self.parameter_handler.declare(
            name="model_reasoning_effort",
            dtype=str,
            default_value=model_reasoning_effort,
            description=f"Reasoning effort spent before generating the completion in {valid_values}.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="completion_parsers",
            dtype=list[str],
            default_value=completion_parsers,
            description="Defines custom parsers that are executed in order after a successful Chat Completion.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="completion_parsers_timeout",
            dtype=float,
            default_value=completion_parsers_timeout,
            description="Time to wait in seconds for each completion parser to terminate.",
            read_only=False,
            range_min=0.0,
            range_max=60.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="completion_parsers_folder",
            dtype=str,
            default_value=completion_parsers_folder,
            description="Path to folder in which completion parsers are looked up first before interpreting them as global paths.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="stream_completion",
            dtype=bool,
            default_value=stream_completion,
            description="Using streaming to receive completions.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="normalize_text_completion",
            dtype=bool,
            default_value=normalize_text_completion,
            description="Applies text normalization to text completions (except JSON mode is used) without affecting the internal state of the context.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="max_tool_calls_per_completion",
            dtype=int,
            default_value=max_tool_calls_per_completion,
            description="A completion that is allowed to contain tool calls must contain at most this many tool calls. Use '0' to deactivate.",
            read_only=False,
            range_min=0,
            range_max=100,
            range_step=1
        )

        self.parameter_handler.declare(
            name="correction_attempts",
            dtype=int,
            default_value=correction_attempts,
            description="Number of self-correction or retry attempts invoked after failed Chat Completions.",
            read_only=False,
            range_min=0,
            range_max=1000,
            range_step=1
        )

        self.parameter_handler.declare(
            name="timeout_chunk_first",
            dtype=float,
            default_value=timeout_chunk_first,
            description="Time in seconds waited until the next Chat Completion chunk is received.",
            read_only=False,
            range_min=0.1,
            range_max=86400.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="timeout_chunk_next",
            dtype=float,
            default_value=timeout_chunk_next,
            description="Time in seconds waited until the first Chat Completion chunk is received.",
            read_only=False,
            range_min=0.1,
            range_max=86400.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="timeout_completion",
            dtype=float,
            default_value=timeout_completion,
            description="Time in seconds waited until a Chat Completion is finished.",
            read_only=False,
            range_min=0.1,
            range_max=86400.0,
            range_step=0.0
        )

        # save defaults

        self.api_endpoints_default = copy.deepcopy(self.api_endpoints)
        self.parameter_defaults = copy.deepcopy(self.parameters.get())

        exclude_defaults_from_reset = ["severity", "log_line_length", "log_last_messages", "log_chunks"]
        for name in exclude_defaults_from_reset:
            del self.parameter_defaults[name]

        # state variables

        self.tools = None
        self.messages = []
        self.awaited_tool_responses = []
        self.is_prompting = False

        # create interfaces

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=50)

        self.cbg_prompt = MutuallyExclusiveCallbackGroup()

        self.srv_prompt = self.create_service(CompletionsPrompt, f"{self.node_namespace}/{self.node_name}/prompt".replace("//", "/"), self.prompt, qos_profile=qos_profile, callback_group=self.cbg_prompt)
        self.srv_interrupt = self.create_service(CompletionsInterrupt, f"{self.node_namespace}/{self.node_name}/interrupt".replace("//", "/"), self.interrupt, qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        self.srv_get_tools = self.create_service(CompletionsToolsGet, f"{self.node_namespace}/{self.node_name}/get_tools".replace("//", "/"), self.get_tools, qos_profile=qos_profile, callback_group=self.cbg_prompt)
        self.srv_set_tools = self.create_service(CompletionsToolsSet, f"{self.node_namespace}/{self.node_name}/set_tools".replace("//", "/"), self.set_tools, qos_profile=qos_profile, callback_group=self.cbg_prompt)

        self.srv_get_context = self.create_service(CompletionsContextGet, f"{self.node_namespace}/{self.node_name}/get_context".replace("//", "/"), self.get_context, qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.srv_set_context = self.create_service(CompletionsContextSet, f"{self.node_namespace}/{self.node_name}/set_context".replace("//", "/"), self.set_context, qos_profile=qos_profile, callback_group=self.cbg_prompt)
        self.srv_reset_parameters = self.create_service(TriggerFeedback, f"{self.node_namespace}/{self.node_name}/reset_parameters".replace("//", "/"), self.reset_parameters, qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        qos_profile_pub = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_ALL, depth=10)
        self.pub_usage = self.create_publisher(String, f"{self.node_namespace}/api_usage".replace("//", "/"), qos_profile=qos_profile_pub, callback_group=MutuallyExclusiveCallbackGroup())

        self.pub_status = self.create_publisher(DiagnosticStatus, f"{self.node_namespace}/{self.node_name}/status".replace("//", "/"), qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.timer_status = self.create_timer(status_interval, self.publish_status, callback_group=MutuallyExclusiveCallbackGroup())

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
            value, message = self.filter_api_endpoint(name, value, self.parameters.log_line_length)

        elif name == "model_reasoning_effort":
            valid_values = ["", "none", "low", "medium", "high"]
            if value not in valid_values:
                message = f"Reasoning effort '{value}' is not in list of unsupported values {valid_values}."
                value = None

        elif name == "completion_parser":
            for parser in value:
                if parser == "":
                    value = None
                    message = "Parser must not be an empty string."
                    break

        if is_declared and self.is_prompting and name in ["api_endpoint", "model_name", "completion_parsers", "stream_completion", "max_tool_calls_per_completion", "correction_attempts"]:
            value = None
            message = "Value cannot be updated while prompting."

        return value, message

    # Utilities

    def validate_tool_properties(self, schema, function_name, path="parameters"):
        if not isinstance(schema, dict):
            return False, f"The function '{function_name}' does not satisfy the required format: '{path}' must be a dict."

        if schema.get('type') != "object":
            return False, f"The function '{function_name}' does not satisfy the required format: '{path}' must be of type 'object'."

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return False, f"The function '{function_name}' does not satisfy the required format: '{path}' is missing a 'properties' dict."

        valid_types = ['boolean', 'string', 'number', 'null', 'object']

        # Validate each property
        for prop_name, prop in properties.items():
            prop_path = f"{path}::properties::{prop_name}"

            if not isinstance(prop, dict):
                return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}' must be of type 'dict'."

            if prop.get('type') == "object":
                # Recursively validate nested object
                ok, msg = self.validate_tool_properties(prop, function_name, path=prop_path)
                if not ok:
                    return False, msg
            else:
                keys = set(prop.keys())
                if not keys.issubset({'type', 'description', 'enum'}):
                    return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}' must only contain 'type', 'description', and optionally 'enum'."

                if 'type' not in prop or 'description' not in prop:
                    return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}' must contain both 'type' and 'description'."

                if prop['type'] not in valid_types:
                    return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}::type' must be in {valid_types}."

                if not isinstance(prop['description'], str):
                    return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}::description' must be a string."

                if 'enum' in prop:
                    if not isinstance(prop['enum'], list):
                        return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}::enum' must be a list."

                    expected_type = str if prop['type'] == "string" else bool if prop['type'] == "boolean" else (int, float)
                    for e in prop['enum']:
                        if not isinstance(e, expected_type):
                            return False, f"The function '{function_name}' does not satisfy the required format: The field '{prop_path}::enum' must only contain elements of type '{expected_type}' instead of '{type(e).__name__}'."

        # Validate 'required'
        if "required" in schema:
            required_list = schema["required"]
            if not isinstance(required_list, list):
                return False, f"The function '{function_name}' does not satisfy the required format: '{path}::required' must be a list."

            for r in required_list:
                if not isinstance(r, str):
                    return False, f"The function '{function_name}' does not satisfy the required format: All items in '{path}::required' must be strings."
                if r not in properties:
                    return False, f"The function '{function_name}' does not satisfy the required format: Required key '{r}' in '{path}::required' is not defined in 'properties'."

        # Validate 'additionalProperties' if 'strict' is true
        if schema.get("strict") is True:
            if "additionalProperties" not in schema:
                return False, f"The function '{function_name}' does not satisfy the required format: '{path}' must include 'additionalProperties' when 'strict' is true."
            if not isinstance(schema["additionalProperties"], bool):
                return False, f"The function '{function_name}' does not satisfy the required format: '{path}::additionalProperties' must be a boolean."
            if schema["additionalProperties"] is True:
                return False, f"The function '{function_name}' does not satisfy the required format: '{path}::additionalProperties' must be False when 'strict' is true."

        return True, ""

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
                    self._logger.warn(f"The context contains a tool response without a corresponding tool call: {message}")
            elif message['role'] == 'assistant':
                if 'tool_calls' in message:
                    for call in message['tool_calls']:
                        assert 'id' in call, f"{call}"
                        all_ids.append(call['id'])
                        awaited_tool_responses.append(call['id'])

        self.awaited_tool_responses = awaited_tool_responses
        self._logger.debug(f"Awaiting tool response{'' if len(self.awaited_tool_responses) == 1 else 's'}: {None if len(self.awaited_tool_responses) == 0 else self.awaited_tool_responses}")

    def check_prompt_validity(self, request):
        response = CompletionsPrompt.Response()
        response.success = False
        response.message = ""
        response.completion = r"{}"

        text = request.text.strip()
        self._logger.debug(f"Received request (role='{request.role}', text='{text}', reset_context='{request.reset_context}', tool_response_id='{request.tool_response_id}', response_type='{request.response_type}')")

        if request.role == 'json':
            try:
                messages = json.loads(text)
            except Exception as e:
                response.message = f"Invalid request - Field 'role' is set to 'json' but 'text' field cannot be parsed as JSON: {repr(e)}"
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response
            else:
                if isinstance(messages, dict):
                    messages = [messages]
                elif not isinstance(messages, list):
                    response.message = f"Invalid request - Provided field 'text' (after JSON-decoding) contains data of invalid type '{type(messages).__name__}'. Supported types are 'dict' (one message) or 'list'."
                    self._logger.error(f"Failed to prompt model: {response.message}")
                    return None, response
                try:
                    for i, message in enumerate(messages):
                        # this ignores awaited_tool_responses for messages i > 0 that are role 'tool'
                        self.check_message_validity(message, ignore_awaited=i > 0)
                except CustomException as e:
                    response.message = f"Invalid request: {e}"
                    self._logger.error(f"Failed to prompt model: {response.message}")
                    return None, response
                try:
                    for i, message in enumerate(messages):
                        messages[i] = self.encode_files(message)
                except CustomException as e:
                    response.message = f"Invalid request: {e}"
                    self._logger.error(f"Failed to prompt model: {response.message}")
                    return None, response

        # role is unknown
        if request.role not in ['system', 'user', 'assistant', 'tool', 'json']:
            response.message = f"Invalid request - Unknown role '{request.role}'. Valid roles are 'system', 'user', 'assistant', 'tool', and 'json'."
            self._logger.error(f"Failed to prompt model: {response.message}")
            return None, response

        if request.role != "json":

            # cannot respond tool call and reset context
            if request.tool_response_id != "" and request.reset_context:
                response.message = f"Invalid request - 'reset_context' cannot be 'True' while 'tool_response_id' is not-empty string '{request.tool_response_id}'."
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response

            # cannot respond to tool call that is not awaited
            if (request.tool_response_id != "" and request.tool_response_id not in self.awaited_tool_responses) and not request.reset_context:
                response.message = f"Invalid request - Not awaiting tool response with ID '{request.tool_response_id}'. Awaiting tool response IDs: '{self.awaited_tool_responses}."
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response

            # enforce response to awaited tool call
            if len(self.awaited_tool_responses) > 0 and not request.reset_context:
                if request.tool_response_id not in self.awaited_tool_responses: # TODO is order important if len(self.awaited_tool_responses) > 1?
                    response.message = f"Invalid request - Awaiting tool response IDs '{self.awaited_tool_responses}'."
                    self._logger.error(f"Failed to prompt model: {response.message}")
                    return None, response

            # tool_response_id is not awaited
            if request.tool_response_id != "" and request.tool_response_id not in self.awaited_tool_responses:
                response.message = f"Invalid request - Unknown tool response ID '{request.tool_response_id}'. Awaiting tool response IDs '{self.awaited_tool_responses}'."
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response

            # tool_response_id must be empty if role is not tool
            if request.tool_response_id != "" and request.role != "tool":
                response.message = f"Invalid request - Tool responses must use role 'tool', not '{request.role}'."
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response

        # enforce response_type 'none', 'text', 'auto' if no tools are defined
        if self.tools is None and request.response_type != "none" and request.response_type != "text" and request.response_type != "json" and request.response_type != "auto":
            response.message = "Invalid request - No tools are defined, so field 'response_type' must be set to 'none', 'text', or 'auto'."
            self._logger.error(f"Failed to prompt model: {response.message}")
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
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response

        # construct messages if role is not 'json'
        if request.role != "json":
            if request.role == "system":
                messages = [{'role': request.role, 'content': text}]
            elif request.role == "user":
                messages = [{'role': request.role, 'content': [{'type': "text", 'text': text}]}]
            elif request.role == "assistant":
                messages = [{'role': request.role, 'content': text}]
            elif request.role == "tool":
                messages = [{'role': request.role, 'tool_call_id': request.tool_response_id, 'content': text}]
            else:
                raise RuntimeError(f"Encountered unexpected role '{request.role}'")
            try:
                self.check_message_validity(messages[-1])
            except CustomException as e:
                response.message = f"Invalid request: {e}"
                self._logger.error(f"Failed to prompt model: {response.message}")
                return None, response

        self._logger.debug("Received request is valid")
        return messages, None

    def check_message_validity(self, message, ignore_awaited=False):
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
                    if element['type'] not in ["text", "image_url", "input_audio", "file"]:
                        raise Exception(f"User message content element type must be in ['text', 'image_url', 'input_audio', 'file'] but it is '{element['type']}'.")
                    if element['type'] == "text":
                        if 'text' not in element:
                            raise Exception("User message content element of type text must contain key 'text'.")
                        if not isinstance(element['text'], str):
                            raise Exception(f"User message content element of type text must contain key 'text' of type 'str' but it is of type '{type(element['text']).__name__}'.")
                        if len(element['text']) == 0:
                            raise Exception("User message content element of type text must contain key 'text' that is not empty.")
                        if not len(element) == 2:
                            raise Exception(f"User message content element of type text must contain exactly two keys 'type' and 'text' but it contains {list(element.keys())}.")
                    elif element['type'] == "image_url":
                        if 'image_url' not in element:
                            raise Exception("User message content element of type image_url must contain key 'image_url'.")
                        if not isinstance(element['image_url'], dict):
                            raise Exception(f"User message content element of type image_url must be of type 'dict' but it is of type {type(element['image_url']).__name__}.")
                        if len(element['image_url']) != 2:
                            raise Exception(f"User message content element of type image_url must contain exactly two keys 'detail' and 'url' but it contains {list(element['image_url'].keys())}.")
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

                    elif element['type'] == "input_audio":
                        if 'input_audio' not in element:
                            raise Exception("User message content element of type input_audio must contain key 'input_audio'.")
                        if not isinstance(element['input_audio'], dict):
                            raise Exception(f"User message content element of type input_audio must be of type 'dict' but it is of type {type(element['input_audio']).__name__}.")
                        if len(element['input_audio']) != 2:
                            raise Exception(f"User message content element of type input_audio must contain exactly two keys 'data' and 'format' but it contains {list(element['input_audio'].keys())}.")
                        if 'data' not in element['input_audio']:
                            raise Exception("User message content element of type input_audio must contain key 'data'.")
                        if not isinstance(element['input_audio']['data'], str):
                            raise Exception(f"User message content element of type input_audio must contain key 'data' of type 'str' but it is of type '{type(element['input_audio']['data']).__name__}'.")
                        if len(element['input_audio']['data']) == 0:
                            raise Exception("User message content element of type input_audio must contain key 'data' that is not empty.")
                        if 'format' not in element['input_audio']:
                            raise Exception("User message content element of type input_audio must contain key 'format'.")
                        if element['input_audio']['format'] not in ["wav", "mp3"]:
                            raise Exception(f"User message content element of type input_audio must contain key 'format' with value in ['wav', 'mp3'] but it is '{element['input_audio']['format']}'.")

                    elif element['type'] == "file":
                        if 'file' not in element:
                            raise Exception("User message content element of type file must contain key 'file'.")
                        if not isinstance(element['file'], dict):
                            raise Exception(f"User message content element of type file must be of type 'dict' but it is of type {type(element['file']).__name__}.")
                        if len(element['file']) != 2:
                            raise Exception(f"User message content element of type file must contain exactly two keys 'filename' and 'file_data' but it contains {list(element['file'].keys())}.")
                        if 'filename' not in element['file']:
                            raise Exception("User message content element of type file must contain key 'filename'.")
                        if not isinstance(element['file']['filename'], str):
                            raise Exception(f"User message content element of type file must contain key 'filename' of type 'str' but it is of type '{type(element['file']['filename']).__name__}'.")
                        if len(element['file']['filename']) == 0:
                            raise Exception("User message content element of type file must contain key 'filename' that is not empty.")
                        if 'file_data' not in element['file']:
                            raise Exception("User message content element of type file must contain key 'file_data'.")
                        if not isinstance(element['file']['file_data'], str):
                            raise Exception(f"User message content element of type file must contain key 'file_data' of type 'str' but it is of type '{type(element['file']['file_data']).__name__}'.")
                        if len(element['file']['file_data']) == 0:
                            raise Exception("User message content element of type file must contain key 'file_data' that is not empty.")

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
                    if element['type'] != 'function':
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
            if not ignore_awaited and message['tool_call_id'] not in self.awaited_tool_responses:
                raise Exception(f"Tool message value of key 'tool_call_id' must be in list of awaited responses {self.awaited_tool_responses} but it is '{message['tool_call_id']}'.")
            if not len(message) == 3:
                raise Exception(f"Tool message must contain exactly three keys 'role', 'content' and 'tool_call_id' but it contains {list(message.keys())}.")

    def encode_files(self, message):
        lut = {
            'image_url': {
                'prefix': "data:image/jpeg;base64,",
                'data': "url"
            },
            'input_audio': {
                'prefix': "",
                'data': "data"
            },
            'file': {
                'prefix': "data:application/pdf;base64,",
                'data': "file_data"
            }
        }

        if message['role'] == "user":
            if isinstance(message['content'], list):
                for i, element in enumerate(message['content']):
                    for modality in lut:
                        if element['type'] == modality:
                            if lut[modality]['prefix'] != "" and element[modality][lut[modality]['data']][:len(lut[modality]['prefix'])] == lut[modality]['prefix']:
                                self._logger.debug(f"Provided '{modality}' is Base64-encoded.")
                            elif is_base64(element[modality][lut[modality]['data']]):
                                self._logger.debug(f"Provided '{modality}' is Base64-encoded without prefix.")
                                message['content'][i][modality][lut[modality]['data']] = f"{lut[modality]['prefix']}{element[modality][lut[modality]['data']]}"
                            elif is_url(element[modality][lut[modality]['data']]):
                                self._logger.debug(f"Provided '{modality}' is a valid URL.")
                            elif os.path.exists(element[modality][lut[modality]['data']]):
                                if os.path.isfile(element[modality][lut[modality]['data']]):
                                    success, _message, b64_encoded = read_as_b64(
                                        file_path=element[modality][lut[modality]['data']],
                                        name=f"file referred to by '{modality}'",
                                        logger=self._logger
                                    )
                                    if success:
                                        message['content'][i][modality][lut[modality]['data']] = f"{lut[modality]['prefix']}{b64_encoded}"
                                    else:
                                        raise CustomException(_message)
                                else:
                                    raise CustomException(f"Provided '{modality}' points to folder '{element[modality][lut[modality]['data']]}'.")
                            else:
                                raise CustomException(f"Provided '{modality}' is neither Base64-encoded, a valid local path, or a web URL.")
                            break
        return message

    def add_request_to_context(self, request, messages):
        if request.reset_context:
            self.messages = []
            self.update_awaited_tool_responses()

        self.message_length_original = len(self.messages)
        self.messages += messages

        # log

        if request.reset_context:
            insert_1 = "Cleared context and added request"
        else:
            insert_1 = "Added request to context"

        if request.response_type == "none":
            insert_2 = "without generating a completion"
        elif request.response_type == "text":
            insert_2 = "before generating a text completion"
        elif request.response_type == "json":
            insert_2 = "before generating json"
        elif request.response_type == "auto":
            insert_2 = "before generating a completion"
        elif request.response_type == "always":
            insert_2 = "before generating tool call"
        else:
            insert_2 = f"before generating a '{request.response_type}' tool call"

        if self.parameters.log_last_messages == 0:
            self._logger.info(f"{insert_1} {insert_2}.")
        else:
            messages_fmt = []

            for i in range(0 if self.parameters.log_last_messages < 0 else max(0, len(self.messages) - self.parameters.log_last_messages), len(self.messages), 1):
                message = self.messages[i]
                if isinstance(message['content'], str):
                    messages_fmt.append(f"{i} - {message['role']}: '{message['content']}'".replace("\n", "\\n"))
                else:
                    for j, content in enumerate(message['content']):
                        insert_3 = f".{j}" if len(message['content']) > 1 else ""
                        if content['type'] == "image_url":
                            messages_fmt.append(f"{i}{insert_3} - {message['role']}: '{'<IMAGE>' if len(content['image_url']['url']) > (self.parameters.log_line_length - len(message['role']) - 7) else content['image_url']['url']}' (detail: '{content['image_url'].get('detail', '')}')")
                        if content['type'] == "input_audio":
                            messages_fmt.append(f"{i}{insert_3} - {message['role']}: '{'<AUDIO>' if len(content['input_audio']['data']) > (self.parameters.log_line_length - len(message['role']) - 7) else content['input_audio']['data']}' (format: '{content['input_audio']['format']}')")
                        if content['type'] == "file":
                            messages_fmt.append(f"{i}{insert_3} - {message['role']}: '{'<FILE>' if len(content['file']['data_url']) > (self.parameters.log_line_length - len(message['role']) - 7) else content['file']['data_url']}' (name: '{content['file']['filename']}')")
                        else:
                            messages_fmt.append(f"{i}{insert_3} - {message['role']}: '{content['text']}'".replace("\n", "\\n"))
            messages_fmt = '\n'.join(messages_fmt)

            log_lines(
                text=f"{insert_1} {insert_2}:\n{messages_fmt}",
                line_length=self.parameters.log_line_length,
                line_highlight="| ",
                block_format=False,
                logger=self._logger,
                severity=20
            )

    def get_no_completion_response(self):
        response = CompletionsPrompt.Response()
        response.success = True
        response.message = "Added request to context without generating a completion."
        response.completion = r"{}"
        return response

    def generate_completion(self, request):
        if self.parameters.probe_api_connection:
            success, message = self.validate_connection(model=self.parameters.model_name)
            if not success:
                response = CompletionsPrompt.Response()
                response.success = False
                response.message = message
                response.completion = r"{}"
                return response

        self.set_tool_choice(request)

        is_valid = True
        corrections = 0
        logs = []

        # do corrections
        while True:
            if not is_valid:
                corrections += 1
                logs.append(f"Starting correction attempt '{corrections}' of '{self.parameters.correction_attempts}'.")
                self._logger.warn(logs[-1])

            self.pipe = multiprocessing.Pipe()
            completion_proc = multiprocessing.Process(target=self.completion_process)
            completion_proc.daemon = True

            reasoning, text, tool_calls, usage, is_complete, stamp_last_chunk = "", "", [], None, False, None
            stamp_start_iso = datetime.datetime.now()
            stamp_start = time.perf_counter()

            completion_proc.start()

            # receive response
            while True:
                now = time.perf_counter()

                if now - stamp_start > self.parameters.timeout_completion:
                    self.pipe[0].send("INTERNAL")
                    usage = self.save_usage(request, None, stamp_start_iso)
                    logs.append(f"Error while receiving completion: Timeout after '{self.parameters.timeout_completion}s' before completion was finished.")
                    break

                if self.parameters.stream_completion is True:
                    if stamp_last_chunk is None:
                        if now - stamp_start > self.parameters.timeout_chunk_first:
                            self.pipe[0].send("INTERNAL")
                            usage = self.save_usage(request, None, stamp_start_iso)
                            logs.append(f"Error while receiving completion: Timeout after '{self.parameters.timeout_chunk_first}s' without receiving the first chunk.")
                            break
                    elif now - stamp_last_chunk > self.parameters.timeout_chunk_next:
                        self.pipe[0].send("INTERNAL")
                        usage = self.save_usage(request, None, stamp_start_iso)
                        logs.append(f"Error while receiving completion: Timeout after '{self.parameters.timeout_chunk_next}s' without receiving the next chunk.")
                        break

                if self.pipe[0].poll():
                    stamp_last_chunk = time.perf_counter()

                    chunk = self.pipe[0].recv()
                    assert isinstance(chunk, dict), f"Expected chunk '{chunk}' to be of type 'dict' instead of '{type(chunk).__name__}'."
                    assert set(chunk.keys()) == {'code', 'content'}, f"Expected chunk '{chunk}' to have keys 'code' and 'content'."
                    assert isinstance(chunk['code'], str), f"Expected chunk code '{chunk['code']}' to be of type 'str' instead of '{type(chunk['code']).__name__}'."
                    assert chunk['code'] in ['INTERRUPT', 'ERROR', 'COMPLETION', 'USAGE', 'ALL_CHUNKS_RECEIVED'], f"Expected chunk code '{chunk['code']}' to be in ['INTERRUPT', 'ERROR', 'COMPLETION', 'USAGE', 'ALL_CHUNKS_RECEIVED']."

                    if chunk['code'] == "ERROR" or chunk['code'] == 'INTERRUPT':
                        assert isinstance(chunk['content'], str), f"Expected chunk content '{chunk['content']}' to be of type 'str' instead of '{type(chunk['content']).__name__}'."

                        if usage is None:
                            usage = self.save_usage(request, None, stamp_start_iso)

                        logs.append(chunk['content'])
                        if len(logs[-1]) > 5000:
                            logs[-1] = logs[-1][:5000] + "..."

                        if chunk['code'] == 'INTERRUPT':
                            response = self.post_process_completion(
                                request=request,
                                reasoning=None,
                                text=None,
                                tool_calls=[],
                                is_valid=False,
                                corrections=None,
                                usage=usage,
                                logs=logs
                            )
                            return response
                        break

                    elif chunk['code'] == "COMPLETION":
                        assert isinstance(chunk['content'], dict), f"Expected chunk content '{chunk['content']}' to be of type 'dict' instead of '{type(chunk['content']).__name__}'."
                        reasoning, text, tool_calls = self.parse_completion_chunk(chunk['content'], reasoning, text, tool_calls)

                    elif chunk['code'] == "USAGE":
                        assert isinstance(chunk['content'], dict), f"Expected chunk content '{chunk['content']}' to be of type 'dict' instead of '{type(chunk['content']).__name__}'."
                        if 'prompt_tokens' not in chunk['content']:
                            logs.append("Ignoring received usage message that misses expected key 'prompt_tokens'.")
                            self._logger.warn(logs[-1])
                        elif 'completion_tokens' not in chunk['content']:
                            logs.append("Ignoring received usage message that misses expected key 'completion_tokens'.")
                            self._logger.warn(logs[-1])
                        else:
                            usage = self.save_usage(request, chunk, stamp_start_iso)

                    elif chunk['code'] == "ALL_CHUNKS_RECEIVED":
                        is_complete = True

                        if usage is None:
                            usage = self.save_usage(request, None, stamp_start_iso)
                            logs.append("Received completion without usage information.")
                            self._logger.warn(logs[-1])

                        # extract tool calls from text
                        if len(tool_calls) == 0 and request.response_type not in ["text", "auto", "json"]:
                            text, tool_calls, logs = self.extract_tool_call_from_text(text, tool_calls, logs)
                        tool_calls, logs = self.clean_tool_call_names(tool_calls, logs)

                        # extract JSON from text
                        if request.response_type == "json":
                            try:
                                json.loads(text)
                            except Exception:
                                dict_extracted = extract_json(text)
                                if dict_extracted is not None:
                                    logs.append(f"Extracted JSON from text completion: '{text}'")
                                    self._logger.warn(logs[-1])
                                    text = json.dumps(dict_extracted)

                        # extract tool call arguments
                        for i in range(len(tool_calls)):
                            try:
                                json.loads(tool_calls[i]['arguments'])
                            except Exception:
                                parameters = extract_json(tool_calls[i]['arguments'])
                                if parameters is not None:
                                    logs.append(f"Extracted JSON from invalid tool call arguments: '{tool_calls[i]['arguments']}'")
                                    self._logger.warn(logs[-1])
                                    tool_calls[i]['arguments'] = json.dumps(parameters)

                        logs = self.add_completion_to_context(reasoning, text, tool_calls, logs)
                        break
                else:
                    time.sleep(0.1)

            if is_complete:
                is_valid, correction_responses, logs = self.check_completion_validity(request, text, tool_calls, logs)
                if is_valid:
                    break
                elif corrections < self.parameters.correction_attempts:
                    log_lines(logs[-1], line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=30)
                    for response in correction_responses:
                        logs.append(f"Adding temporary correction message to context: '{response}'")
                        self._logger.debug(logs[-1])
                    self.messages += correction_responses
                    logs.append(f"Attempt failed after '{time.perf_counter() - stamp_start:.1f}s'.")
                    self._logger.debug(logs[-1])
                else:
                    log_lines(logs[-1], line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=40)
                    logs.append(f"Attempt failed after '{time.perf_counter() - stamp_start:.1f}s'.")
                    self._logger.debug(logs[-1])
                    reasoning, text, tool_calls = "", "", []
                    break
            else:
                is_valid = False
                if corrections < self.parameters.correction_attempts:
                    log_lines(logs[-1], line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=30)
                    logs.append(f"Attempt failed after '{time.perf_counter() - stamp_start:.1f}s'.")
                    self._logger.debug(logs[-1])
                else:
                    log_lines(logs[-1], line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=40)
                    logs.append(f"Attempt failed after '{time.perf_counter() - stamp_start:.1f}s'.")
                    self._logger.debug(logs[-1])
                    reasoning, text, tool_calls = "", "", []
                    break

        response = self.post_process_completion(
            request=request,
            reasoning=reasoning if len(reasoning) > 0 else None,
            text=text if len(text) > 0 else None,
            tool_calls=tool_calls,
            is_valid=is_valid,
            corrections=corrections,
            usage=usage,
            logs=logs
        )
        return response

    def set_tool_choice(self, request):
        if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "openai":
            if request.response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {'type': "json_object"}
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "required"
            elif request.response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': request.response_type}}

        elif self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "mistral":
            if request.response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {'type': "json_object"}
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "any"
            elif request.response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': request.response_type}}

        elif self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "openrouter":
            if request.response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {'type': "json_object"}
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "required"
                # self.response_format = self.tools_to_response_format(self.tools)
                # self.tool_choice = "auto"
            elif request.response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': request.response_type}}

        elif self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "vllm":
            if request.response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif request.response_type == "json":
                self.response_format = {'type': "json_object"}
                # self.response_format = {'type': "text"} # set this to deactivate JSON-mode; response with invalid JSON will still trigger self-correctionnnnnnnnnnn.
                self.tool_choice = "none"
            elif request.response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto" # wait until v1 engines supports 'required'
                self._logger.warn(f"Tool choice '{request.response_type}' is not available for api_flavor '{self.api_endpoints[self.parameters.api_endpoint]['api_flavor']}', using '{self.tool_choice}' instead")
            elif request.response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': request.response_type}}

        else:
            raise NotImplementedError(f"Undefined API flavor '{self.api_endpoints[self.parameters.api_endpoint]['api_flavor']}'.")

    def tools_to_response_format(self, tools):
        schemas = []
        for tool in tools:
            fn = tool['function']
            name = fn['name']
            schema = fn['parameters']
            schema["additionalProperties"] = False

            schemas.append({
                'name': name,
                "strict": True,
                "schema": schema
            })

        return {
            'type': "json_schema",
            "json_schema": schemas[0] if len(schemas) == 1 else schemas
        }

    def completion_process(self):
        self._logger.debug("completion_process(): start")

        success, message, api_key = self.retrieve_api_key()
        if not success:
            self.pipe[1].send({'code': "ERROR", 'content': message})
            return

        messages = copy.deepcopy(self.messages)

        # condense consecutive user messages
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
                    self._logger.debug(f"Condensing '{len(contents)}' consecutive user messages ('{first}' to '{last}') into a single one.")
                    new_message = {'role': "user", 'content': contents}
                    messages = messages[: first] + [new_message] + messages[last + 1:]
                    break
                elif messages[i]['role'] != "user":
                    is_user = 0
            else:
                break

        messages_print = copy.deepcopy(messages)
        for i, message in enumerate(messages_print):
            if messages[i]['role'] == "user":
                if isinstance(message['content'], list):
                    for j, element in enumerate(message['content']):
                        if element['type'] == "image_url":
                            if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "vllm":
                                self._logger.debug(f"Tepmorarily stripping image detail '{messages[i]['content'][j]['image_url']['detail']}' for using vLLM")
                                del messages[i]['content'][j]['image_url']['detail']
                                del messages_print[i]['content'][j]['image_url']['detail']
                            messages_print[i]['content'][j]['image_url']['url'] = "<IMAGE>"
                        elif element['type'] == "input_audio":
                            # if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "mistral":
                            #     self._logger.debug(f"Tepmorarily stripping audio format '{messages[i]['content'][j]['input_audio']['format']}' for using Mistral AI")
                            #     del messages[i]['content'][j]['input_audio']['format']
                            #     del messages_print[i]['content'][j]['input_audio']['format']
                            messages_print[i]['content'][j]['input_audio']['data'] = "<AUDIO>"
                        elif element['type'] == "file":
                            messages_print[i]['content'][j]['file']['file_data'] = "<FILE>"
            messages_print[i] = f"{i}: " + str(messages_print[i]).replace("\n", "\\n")

        log_lines("Context:\n" + str('\n'.join(messages_print)), line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=10)

        try:
            headers = {
                'Authorization': f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/AIS-Bonn/nimbro_api",
                "X-Title": "NimbRo-API",
                'Content-Type': "application/json",
            }

            if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "openai":
                data = {
                    'model': self.parameters.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'temperature': self.parameters.model_temperature,
                    'top_p': self.parameters.model_top_p,
                    'max_completion_tokens': self.parameters.model_max_tokens,
                    'presence_penalty': self.parameters.model_presence_penalty,
                    'frequency_penalty': self.parameters.model_frequency_penalty,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self.parameters.stream_completion
                }
                if self.parameters.model_reasoning_effort not in ["", "none"]:
                    data['reasoning_effort'] = self.parameters.model_reasoning_effort
                if self.tools is not None:
                    data['tool_choice'] = self.tool_choice
                    if self.parameters.model_name[0] != "o":
                        data['parallel_tool_calls'] = self.parameters.max_tool_calls_per_completion > 1
                if self.parameters.stream_completion is True:
                    data['stream_options'] = {'include_usage': True}

            elif self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "mistral":
                data = {
                    'model': self.parameters.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'tool_choice': self.tool_choice,
                    'temperature': self.parameters.model_temperature,
                    'top_p': self.parameters.model_top_p,
                    'max_tokens': self.parameters.model_max_tokens,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self.parameters.stream_completion
                }

            elif self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "openrouter":
                data = {
                    'model': self.parameters.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'temperature': self.parameters.model_temperature,
                    'top_p': self.parameters.model_top_p,
                    'max_tokens': self.parameters.model_max_tokens,
                    'presence_penalty': self.parameters.model_presence_penalty,
                    'frequency_penalty': self.parameters.model_frequency_penalty,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self.parameters.stream_completion
                }
                if self.parameters.model_reasoning_effort not in ["", "none"]:
                    data['reasoning'] = {
                        'effort': self.parameters.model_reasoning_effort,
                        'exclude': False
                    }
                if self.tools is not None:
                    data['tool_choice'] = self.tool_choice
                if self.parameters.stream_completion is True:
                    data['stream_options'] = {'include_usage': True}

            elif self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] == "vllm":
                data = {
                    'model': self.parameters.model_name,
                    'messages': messages,
                    'tools': self.tools,
                    'temperature': self.parameters.model_temperature,
                    'top_p': self.parameters.model_top_p,
                    'max_tokens': self.parameters.model_max_tokens,
                    'presence_penalty': self.parameters.model_presence_penalty,
                    'frequency_penalty': self.parameters.model_frequency_penalty,
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self.parameters.stream_completion,
                    'chat_template_kwargs': {'enable_thinking': self.parameters.model_reasoning_effort not in ["", "none"]}
                }
                if self.tools is not None:
                    data['tool_choice'] = self.tool_choice
                    data['parallel_tool_calls'] = self.parameters.max_tool_calls_per_completion > 1
                if self.parameters.stream_completion is True:
                    data['stream_options'] = {'include_usage': True}

            else:
                message = f"Undefined API flavor '{self.api_endpoints[self.parameters.api_endpoint]['api_flavor']}'."
                self.pipe[1].send({'code': "ERROR", 'content': message})
                return

            self._logger.debug("Sending POST request")
            completion = requests.post(self.api_endpoints[self.parameters.api_endpoint]['completions_url'], headers=headers, json=data, stream=self.parameters.stream_completion)

            if not self.parameters.stream_completion:
                if completion.status_code != 200:
                    message = f"Received unexpected HTTP status code '{completion.status_code}': {completion.text}"
                    message = remove_whitespace(string=message, reduce_to_single_space=True)
                    self.pipe[1].send({'code': "ERROR", 'content': message})
                else:
                    try:
                        json_data = completion.json()
                    except Exception as e:
                        message = f"Error while receiving completion: Failed to parse POST response as JSON: {repr(e)}"
                        self.pipe[1].send({'code': "ERROR", 'content': message})
                    else:
                        log_lines(f"POST response:\n{json.dumps(json_data, indent=2)}", line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, allow_empty_lines=True, logger=self._logger, severity=10)

                        # usage
                        if 'usage' in json_data:
                            self.pipe[1].send({'code': "USAGE", 'content': json_data['usage']})
                        # choices
                        if 'choices' not in json_data:
                            message = "Error while receiving completion: Expected POST response to contain key 'choices'."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif not isinstance(json_data['choices'], list):
                            message = f"Error while receiving completion: Expected value of key 'choices' to be of type 'list' instead of '{type(json_data['choices']).__name__}'."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif len(json_data['choices']) == 0:
                            message = "Error while receiving completion: Expected list 'choices' to contain at least one element."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        # finish_reason
                        elif 'finish_reason' not in json_data['choices'][0]:
                            message = "Error while receiving completion: Expected choice to contain key 'finish_reason'."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif json_data['choices'][0]['finish_reason'] not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                            message = f"Error while receiving completion: Expected value of key 'finish_reason' to be in '{[None, 'stop', 'tool_calls', 'STOP', 'end_turn']}' instead of '{json_data['choices'][0]['finish_reason']}'."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        # message
                        elif 'message' not in json_data['choices'][0]:
                            message = "Error while receiving completion: Expected choice to contain key 'message'."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        elif not isinstance(json_data['choices'][0]['message'], dict):
                            message = f"Error while receiving completion: Expected value of key 'message' to be of type 'dict' instead of '{type(json_data['choices'][0]['message']).__name__}'."
                            self.pipe[1].send({'code': "ERROR", 'content': message})
                        else:
                            completion = json_data['choices'][0]['message']
                            if 'tool_calls' in completion and completion['tool_calls'] is None:
                                del completion['tool_calls']
                            self.pipe[1].send({'code': "COMPLETION", 'content': completion})
                            self.pipe[1].send({'code': "ALL_CHUNKS_RECEIVED", 'content': ''})

                self._logger.debug("completion_process(): end")
                return

        except Exception as e:
            message = f"Failed to POST request: {repr(e)}"
            self.pipe[1].send({'code': "ERROR", 'content': message})

        else:
            self._logger.debug("Sent POST request")

            decoded_buffer = ""
            undecoded_buffer = b""
            error = ""
            early_stop = False
            usage = None

            for chunk in completion.iter_content(chunk_size=1):
                # self._logger.debug(f"chunk: {chunk}")
                if early_stop is True:
                    break
                decoded = False

                # check if response was canceled from external source
                if self.pipe[1].poll():
                    code = self.pipe[1].recv()
                    if code == "EXTERNAL":
                        message = "Completion was interrupted due to request from external source."
                        self._logger.debug(message)
                        self.pipe[1].send({'code': "INTERRUPT", 'content': message})
                    else:
                        self._logger.debug("Completion was interrupted due to request from internal source")
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
                                self._logger.warn(f"Ignoring byte sequence '{undecoded_buffer}' after failure to decode it")
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
                    # self._logger.debug(f"{"\n" in decoded_buffer}: decoded_buffer: {decoded_buffer.replace("\n", "\\n")}")
                    while '\n' in decoded_buffer:
                        line, decoded_buffer = decoded_buffer.split('\n', 1)
                        # self._logger.debug(f"line: {line}")
                        if line != "":
                            if line.find('data:') == 0:

                                # end of response
                                if line == 'data: [DONE]':
                                    # forward usage before end of process
                                    if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                        if usage is None:
                                            self._logger.warn("Did not receive usage before [DONE] message")
                                        else:
                                            self.pipe[1].send({'code': "USAGE", 'content': usage})
                                    self._logger.debug("Received [DONE] message")
                                    self.pipe[1].send({'code': "ALL_CHUNKS_RECEIVED", 'content': ''})
                                else:
                                    try:
                                        json_data = json.loads(line[6:])
                                    except Exception as e:
                                        self._logger.warn(f"Ignoring line '{line}' after failure to parse it as JSON: {repr(e)}")
                                    else:
                                        # unexpected finish reason
                                        if json_data.get('finish_reason') not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                                            # forward usage before end of process
                                            if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                                if usage is None:
                                                    self._logger.warn("Did not receive usage before [ERROR] message")
                                                else:
                                                    self.pipe[1].send({'code': "USAGE", 'content': usage})
                                            message = f"Error while receiving completion: Unexpected finish reason '{json_data.get('finish_reason')}'."
                                            self.pipe[1].send({'code': "ERROR", 'content': message})
                                            early_stop = True
                                            break
                                        else:
                                            # extract usage
                                            if json_data.get('usage') is not None:
                                                if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                                    usage = json_data['usage']
                                                else:
                                                    self.pipe[1].send({'code': "USAGE", 'content': json_data['usage']})

                                            # extract choices
                                            if len(json_data.get('choices', [])) > 0:
                                                try:
                                                    json_choice = json_data['choices'][0]
                                                except Exception as e:
                                                    self._logger.warn(f"Ignoring data '{json_data}' after failure to parse choice as JSON: {repr(e)}")
                                                else:
                                                    # unexpected finish reason
                                                    if json_choice.get('finish_reason') not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                                                        # forward usage before end of process
                                                        if self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                                                            if usage is None:
                                                                self._logger.warn("Did not receive usage before [ERROR] message")
                                                            else:
                                                                self.pipe[1].send({'code': "USAGE", 'content': usage})
                                                        message = f"Error while receiving completion: Unexpected finish reason '{json_choice.get('finish_reason')}'."
                                                        self.pipe[1].send({'code': "ERROR", 'content': message})
                                                        early_stop = True
                                                        break
                                                    else:
                                                        # forward delta
                                                        self.pipe[1].send({'code': "COMPLETION", 'content': json_choice['delta']})
                            else:
                                error += line
            else:
                self._logger.debug("Received full POST response")

                if len(undecoded_buffer) > 0:
                    self._logger.warn(f"Ignoring byte sequence '{undecoded_buffer}' after failure to decode it")

                if len(decoded_buffer) > 0:
                    error += decoded_buffer

                # forward remaining usage before end of process
                if usage is not None and self.api_endpoints[self.parameters.api_endpoint]['api_flavor'] in ["vllm", "openrouter"]:
                    self.pipe[1].send({'code': "USAGE", 'content': usage})

                # forward collected error
                if error != "":
                    message = f"Error while receiving completion: {error}."
                    message = remove_whitespace(string=message, reduce_to_single_space=True)
                    self.pipe[1].send({'code': "ERROR", 'content': message})

            completion.close()
            self._logger.debug("Connection closed")

        self._logger.debug("completion_process(): end")

    def save_usage(self, request, chunk, stamp_start_iso):
        stamp_stop = datetime.datetime.now()

        usage = {}
        usage['api_type'] = "completions"
        usage['api_endpoint'] = self.parameters.api_endpoint
        usage['model_name'] = self.parameters.model_name
        if request.identifier != "":
            usage['identifier'] = request.identifier
        usage['stamp_start'] = convert_stamp(stamp=stamp_start_iso, target_format="iso")
        usage['stamp_stop'] = convert_stamp(stamp=stamp_stop, target_format="iso")
        usage['duration'] = (stamp_stop - stamp_start_iso).total_seconds()

        if chunk is not None:
            # Ignoring everything other than 'prompt_tokens', 'cached_tokens', and 'completion_tokens' until provers agree on a standard to deal with reasoning/audio/image tokens.
            if chunk['content']['prompt_tokens'] > 0:
                usage['tokens_input_uncached'] = chunk['content']['prompt_tokens']
            if 'prompt_tokens_details' in chunk['content'] and chunk['content']['prompt_tokens_details'] is None:
                del chunk['content']['prompt_tokens_details']
            cashed = chunk['content'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
            if cashed > 0:
                usage['tokens_input_cached'] = cashed
            if chunk['content']['completion_tokens'] > 0:
                usage['tokens_output'] = chunk['content']['completion_tokens']

            # def clean_dict(d):
            #     cleaned = {}
            #     for key, value in d.items():
            #         if isinstance(value, dict):
            #             sub = clean_dict(value)
            #             if sub:
            #                 cleaned[key] = sub
            #         elif value is not None and value != 0:
            #             cleaned[key] = value
            #     return cleaned
            # tokens = clean_dict(chunk['content'])
            # for key in tokens:
            #     assert key not in usage, f"{key}"
            #     usage[key] = tokens[key]

        usage_str = json.dumps(usage, indent=2)
        log_lines(f"Usage:\n{usage_str}", line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=10)

        usage_msg = String()
        usage_msg.data = usage_str
        self.pub_usage.publish(usage_msg)

        return usage

    def parse_completion_chunk(self, chunk, reasoning, text, tool_calls):
        if self.parameters.log_chunks:
            log_lines(f"Received chunk:\n{json.dumps(chunk, indent=4)}", line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, allow_empty_lines=True, logger=self._logger, severity=10)

        # chunk contains reasoning
        for key in ['reasoning', 'reasoning_content']:
            if chunk.get(key) not in ["", None]:
                if self.parameters.log_chunks:
                    self._logger.debug(f"Chunk contains '{key}'")
                if not isinstance(chunk[key], str):
                    raise AssertionError(f"Expected value of key '{key}' to be of type 'str' instead of '{type(chunk['key']).__name__}': {chunk}")
                reasoning += chunk[key]

        # chunk contains text
        if chunk.get('content') not in ["", None]:
            if self.parameters.log_chunks:
                self._logger.debug("Chunk contains 'content'")
            if not isinstance(chunk['content'], str):
                raise AssertionError(f"Expected value of key 'content' to be of type 'str' instead of '{type(chunk['content']).__name__}': {chunk}")
            text += chunk['content']

        # chunk contains tool call
        if chunk.get('tool_calls') not in [[], "", None]:
            if self.parameters.log_chunks:
                self._logger.debug("Chunk contains 'tool_calls'")
            if not isinstance(chunk['tool_calls'], list):
                raise AssertionError(f"Expected value of key 'tool_calls' to be of type 'list' instead of '{type(chunk['tool_calls']).__name__}': {chunk}")
            if self.parameters.log_chunks:
                self._logger.debug(f"Chunk contains '{len(chunk['tool_calls'])}' toll call{'' if len(chunk['tool_calls']) == 1 else 's'}")
            for i in range(len(chunk['tool_calls'])):
                if self.parameters.log_chunks:
                    self._logger.debug(f"Handling tool call '{i}'")
                if isinstance(chunk['tool_calls'][i].get('index'), int) and isinstance(chunk['tool_calls'][i].get('function'), dict):
                    if isinstance(chunk['tool_calls'][i].get('id'), str) and len(chunk['tool_calls'][i]['id']) > 0 and isinstance(chunk['tool_calls'][i]['function'].get('name'), str) and len(chunk['tool_calls'][i]['function']['name']) > 0:
                        if len(tool_calls) == chunk['tool_calls'][i]['index']:
                            if self.parameters.log_chunks:
                                self._logger.debug(f"Appending new tool call with index '{chunk['tool_calls'][i]['index']}'")
                            tool_calls.append({'id': chunk['tool_calls'][i]['id'], 'name': chunk['tool_calls'][i]['function']['name'], 'arguments': ""})
                        else:
                            raise AssertionError(f"Expected value of key 'index' in value of key 'tool_calls' to be '{len(tool_calls)}' instead of '{chunk['tool_calls'][i]['index']}'.")
                    if chunk['tool_calls'][i]['function'].get('arguments') not in ["", None]:
                        if chunk['tool_calls'][i]['index'] < len(tool_calls):
                            if self.parameters.log_chunks:
                                self._logger.debug(f"Appending arguments to tool call with index '{chunk['tool_calls'][i]['index']}'")
                            tool_calls[chunk['tool_calls'][i]['index']]['arguments'] += chunk['tool_calls'][i]['function']['arguments']
                        else:
                            raise AssertionError(f"Expected value of key 'index' in value of key 'tool_calls' to be smaller '{len(tool_calls)}' instead of '{chunk['tool_calls'][i]['index']}'.")
                elif set(chunk['tool_calls'][i].keys()) == {'id', 'type', 'function'}:
                    assert isinstance(chunk['tool_calls'][i]['id'], str) and len(chunk['tool_calls'][i]['id']) > 0, f"Expected value of key 'id' in value of key 'tool_calls' to be a non-empty string instead of '{chunk['tool_calls'][i]['id']}'."
                    assert chunk['tool_calls'][i]['type'] == "function", f"Expected value of key 'id' in value of key 'tool_calls' to be 'function' instead of '{chunk['tool_calls'][i]['type']}'."
                    assert isinstance(chunk['tool_calls'][i]['function'], dict), f"Expected value of key 'id' in value of key 'tool_calls' to be dictionary instead of '{chunk['tool_calls'][i]['function']}'."
                    assert isinstance(chunk['tool_calls'][i]['function']['name'], str) and len(chunk['tool_calls'][i]['function']['name']) > 0, f"Expected value of key 'name' in value of key 'function' to be a non-empty string instead of '{chunk['tool_calls'][i]['function']['name']}'."
                    assert isinstance(chunk['tool_calls'][i]['function']['arguments'], str), f"Expected value of key 'arguments' in value of key 'function' to be a non-empty string instead of '{chunk['tool_calls'][i]['function']['arguments']}'."
                    tool_calls.append({'id': chunk['tool_calls'][i]['id'], 'name': chunk['tool_calls'][i]['function']['name'], 'arguments': chunk['tool_calls'][i]['function']['arguments']})
                else:
                    raise AssertionError(f"Expected value of key 'tool_calls' to either contain the fields 'index' and 'function', or 'id', 'type' and 'function', instead of {list(chunk['tool_calls'][i].keys())}.")

        return reasoning, text, tool_calls

    def extract_tool_call_from_text(self, text, tool_calls, logs):
        first_text_call = extract_json(text, first_over_longest=True)
        if first_text_call is not None:
            if 'name' in first_text_call and 'arguments' in first_text_call:
                logs.append("Extracted tool call from text completion and moved it to tool calls.")
                self._logger.warn(logs[-1])
                text = text.replace(first_text_call, "")
                if 'id' in first_text_call:
                    first_text_call['id'] = first_text_call['id']
                else:
                    logs.append("Generating missing ID for extracted tool call.")
                    self._logger.warn(logs[-1])
                    made_up_id = self.get_clock().now().seconds_nanoseconds()
                    made_up_id = f"{made_up_id[0]}_{made_up_id[1]}"
                    first_text_call['id'] = made_up_id
                first_text_call['arguments'] = json.dumps(first_text_call['arguments'])
                tool_calls.append(first_text_call)

        return text, tool_calls, logs

    def clean_tool_call_names(self, tool_calls, logs):
        # I experienced openai referring to undefined functions names in a way that includes special characters (e.g. 'assistant.tell_joke' instead of 'tell_joke').
        # Responding to such a function would cause the completion to respond with 'invalid function name' due to the illegal use of special characters.
        # So, we remove special characters here, establish a legal function name, and then let the self correction routines check validity w.r.t. the defined JSON Schema.
        for i, call in enumerate(tool_calls):
            if not re.match('^[a-zA-Z0-9_-]{1,64}$', call['name']):
                tool_calls[i]['name'] = re.sub(r"[^a-zA-Z0-9_-]", "", call['name'])
                tool_calls[i]['name'] = tool_calls[i]['name'][:64]
                logs.append(f"Renaming tool call with invalid name '{call['name']}' to '{tool_calls[i]['name']}'.")
                self._logger.warn(logs[-1])
        return tool_calls, logs

    def add_completion_to_context(self, reasoning, text, tool_calls, logs):
        def print_unvalidated_tool(dictionary):
            dictionary = copy.deepcopy(dictionary)
            try:
                dictionary['arguments'] = json.loads(dictionary['arguments'])
            except Exception:
                pass
            return json.dumps(dictionary, indent=2)

        if sum([reasoning != "", text != "", len(tool_calls) > 0]) > 1:
            response_msg = "Mixed completion:\n"
            if reasoning != "":
                response_msg += f"\n{escape['bold']}{escape['underline']}Reasoning{escape['end']}:\n'\n{reasoning}\n'\n"
            if text != "":
                response_msg += f"\n{escape['bold']}{escape['underline']}Text{escape['end']}:\n'\n{text}\n'\n"
            if len(tool_calls) > 0:
                tool_msg = ',\n'.join([f"{(i + ': ') if len(tool_calls) > 1 else ''}{print_unvalidated_tool(tool)}" for i, tool in enumerate(tool_calls)])
                if len(tool_calls) > 1:
                    tool_msg = f"[{tool_msg}]"
                response_msg += f"\n{escape['bold']}{escape['underline']}Tool call{'' if len(tool_calls) == 1 else 's'}{escape['end']}:\n{tool_msg}\n"
        elif reasoning != "":
            response_msg = f"{escape['bold']}{escape['underline']}Reasoning completion{escape['end']}:\n'\n{reasoning}\n'"
        elif text != "":
            response_msg = f"{escape['bold']}{escape['underline']}Text completion{escape['end']}:\n'\n{text}\n'"
        elif len(tool_calls) > 0:
            tool_msg = '\n\n'.join([f"{(i + ': ') if len(tool_calls) > 1 else ''}{print_unvalidated_tool(tool)}" for i, tool in enumerate(tool_calls)])
            response_msg = f"{escape['bold']}{escape['underline']}Tool call{'' if len(tool_calls) == 1 else 's'}{escape['end']}:\n{tool_msg}"
        else:
            response_msg = f"Malformed completion:\nReasoning: {reasoning}\nText: {text}\nTool calls: {tool_calls}\n"
        log_lines(response_msg, line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, allow_empty_lines=True, logger=self._logger, severity=20)

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
            logs.append(f"Unexpected error in validity check of completion '{message}': {e}")
            self._logger.error(logs[-1])

        self.messages.append(message)

        log_lines(f"Completion added to context:\n{json.dumps(message, indent=2)}", line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, logger=self._logger, severity=10)

        return logs

    def check_completion_validity(self, request, text, tool_calls, logs):
        is_valid = True

        # create generic correction response # TODO have a single correction response rather than number-of-tools-calls + 1 single messages

        correction_responses = []
        tool_call_is_valid_default_correction = "This tool call is valid and does not require any correction."
        for i, call in enumerate(tool_calls):
            correction_responses.append({})
            correction_responses[-1]['role'] = "tool"
            correction_responses[-1]['tool_call_id'] = call['id']
            correction_responses[-1]['content'] = tool_call_is_valid_default_correction
        correction_responses.append({})
        correction_responses[-1]['role'] = "user"
        correction_responses[-1]['content'] = "Your response is invalid. Please correct it based on the provided error messages and try again!"

        # test error cases

        # error case: tool use when there should not be any tool use
        if (self.tools is None or request.response_type == "text") and len(tool_calls) > 0:
            is_valid = False
            logs.append(f"Completion contains a tool call despite {'no tools being defined' if self.tools is None else 'only text was requested'}.")
            for i in range(len(correction_responses)):
                if 'tool_call_id' in correction_responses[i]:
                    correction_responses[i]['content'] = "Your response must not contain any tool call, but only text content."

        # error case: tool choice "use specific function" was violated
        if self.tools is not None and request.response_type != "text" and request.response_type != "auto" and request.response_type != "always" and request.response_type != "json":
            if text != "":
                is_valid = False
                logs.append(f"Completion contains text content despite tool choice being set to '{request.response_type}'.")
                correction_responses[-1]['content'] = f"Your response must only contain a tool call of '{request.response_type}' without additional text."
            else:
                valid_ids = []
                invalid_ids_names = {}
                for c in tool_calls:
                    if c['name'] == request.response_type:
                        valid_ids.append(c['id'])
                    else:
                        invalid_ids_names[c['id']] = c['name']
                for i in range(len(correction_responses)):
                    if correction_responses[i]['role'] == "tool":
                        if not correction_responses[i]['tool_call_id'] in valid_ids:
                            is_valid = False
                            logs.append(f"Completion contains tool call '{invalid_ids_names[correction_responses[i]['tool_call_id']]}' despite tool choice being set to '{request.response_type}'.")
                            correction_responses[i]['content'] = f"Your response must only contain the tool call '{request.response_type}'."

        # error case: exceeding maximum number of tool calls per response
        if len(tool_calls) > self.parameters.max_tool_calls_per_completion and self.parameters.max_tool_calls_per_completion > 0:
            is_valid = False
            logs.append(f"Completion contains '{len(tool_calls)}' tool calls, but the maximum number of tool calls per completion is '{self.parameters.max_tool_calls_per_completion}'.")
            for i in range(len(correction_responses)):
                if 'tool_call_id' in correction_responses[i]:
                    correction_responses[i]['content'] = f"Your response must contain at most {self.parameters.max_tool_calls_per_completion} tool call{'' if self.parameters.max_tool_calls_per_completion == 1 else 's'}, but yours contains {len(tool_calls)} tool calls. Please filter accordingly and try again!"

        # error case: custom tool choice "always" was violated
        if request.response_type == "always" and len(tool_calls) == 0:
            is_valid = False
            logs.append("Completion does not contain a tool call despite tool choice being set to value 'always'.")
            for i in range(len(correction_responses)):
                if 'tool_call_id' not in correction_responses[i]:
                    if self.parameters.max_tool_calls_per_completion == 1:
                        correction_responses[i]['content'] = "Please express your last message in a tool call instead of a text response!"
                    else:
                        correction_responses[i]['content'] = f"Your response must contain {'at least one' if self.parameters.max_tool_calls_per_completion > 1 else 'a'} tool call. Please try again!"

        # error case: function call violates JSON Schema
        for i, call in enumerate(tool_calls):
            for j in range(len(correction_responses)):
                if 'tool_call_id' in correction_responses[j]:
                    if call['id'] == correction_responses[j]['tool_call_id']:
                        if correction_responses[j]['content'] == tool_call_is_valid_default_correction:
                            valid, reason, logs = self.validate_tool_call(call, logs)
                            if not valid:
                                is_valid = False
                                correction_responses[j]['content'] = reason
                        else:
                            self._logger.debug(f"Skipping JSON Schema based validity check of tool call '{call['name']}' as it is already considered invalid by some previous filter")

        # error case: text response cannot be parsed as JSON despite JSON-mode being activated
        if request.response_type == "json":
            try:
                json.loads(text)
            except Exception as e:
                is_valid = False
                logs.append(f"Text completion cannot be parsed as JSON despite completion type being set to JSON: {repr(e)}")
                correction_responses[-1]['content'] = "Your response cannot be parsed as JSON. Please try again and respond only with valid JSON and no additional text."
            else:
                self._logger.debug("Text completion parses as JSON")

        return is_valid, correction_responses, logs

    def validate_tool_call(self, tool_call, logs):
        success = True
        reason = None

        call_name = tool_call.get('name')
        call_args = tool_call.get('arguments')

        if call_name is None or call_args is None:
            success = False
            logs.append("Completion contains a tool call that that misses the keys 'name' and/or 'arguments'.")
            reason = "Your response contains an invalid tool call. A tool call must contain the keys 'name' and 'arguments'."
        else:
            matched = next(
                (tool for tool in self.tools if tool.get('type') == "function" and tool.get("function", {}).get('name') == call_name),
                None
            )

            if matched is None:
                success = False
                logs.append("Completion contains a tool call that cannot be associated with any defined tool.")
                reason = "Your response contains a tool call that cannot be associated with any of the defined tools."
            else:
                schema = matched["function"].get("parameters", {})
                try:
                    arguments = json.loads(call_args)
                except json.JSONDecodeError as e:
                    success = False
                    logs.append(f"Completion contains a tool call with key 'arguments' that cannot be parsed as JSON: {e.msg}")
                    reason = f"Your response contains a tool call of which the arguments cannot be parsed as JSON: {e.msg}"
                else:
                    if JSONSCHEMA_AVAILABLE:
                        validator = jsonschema.Draft7Validator(schema)
                        errors = sorted(validator.iter_errors(arguments), key=lambda e: e.path)
                        if errors:
                            success = False
                            logs.append(f"Completion contains a tool call that violates the JSON Schema: {errors[0].message}")
                            reason = f"Your response contains a tool call that violates the JSON Schema: {errors[0].message}"
                    else:
                        logs.append("Tool call cannot be validated against tool definitions because the 'jsonschema' module is not available.")
                        self._logger.warn(logs[-1], once=True)

        if success:
            self._logger.debug("Tool call is valid")

        return success, reason, logs

    def execute_parser(self, executable_path, success, message, completion, timeout):
        if executable_path == "":
            return success, message, completion

        response_dict = {
            'success': success,
            'message': message,
            'completion': completion
        }
        response_json = json.dumps(response_dict)

        path_with_prefix = os.path.join(self.parameters.completion_parsers_folder, executable_path)
        if os.path.isfile(path_with_prefix):
            self._logger.debug(f"Parser path '{executable_path}' points to parser in '{self.parameters.completion_parsers_folder}'")
            executable_path = path_with_prefix
        self._logger.debug(f"Executing parser '{executable_path}'")

        succeeded = False

        try:
            if not os.path.isfile(executable_path):
                raise CustomException(f"Parser '{executable_path}' does not exist.")

            result = subprocess.run(
                [executable_path],
                input=response_json,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout
            )

            parser_logs = result.stderr.strip()
            if parser_logs != "":
                self._logger.info(f"Parser '{executable_path}': {result.stderr.strip()}")

            if result.returncode != 0:
                raise CustomException(f"Parser '{executable_path}' exited with non-zero status '{result.returncode}'.")

            output_str = result.stdout.strip()
            if output_str == "":
                raise CustomException(f"Parser '{executable_path}' exited without returning output.")

            try:
                processed_dict = json.loads(output_str)
            except Exception as e:
                raise CustomException(f"Parser '{executable_path}' returned string '{output_str}' that cannot be parsed as JSON: {repr(e)}")

            if not isinstance(processed_dict, dict):
                raise CustomException(f"Parser '{executable_path}' returned JSON that parses as '{type(response_dict).__name__}' instead of 'dict'.")

            if set(processed_dict.keys()) != set(response_dict.keys()):
                raise CustomException(f"Parser '{executable_path}' returned JSON dictionary with unexpected keys '{list(processed_dict.keys())}' instead of '{list(response_dict.keys())}'.")

            if not isinstance(processed_dict['success'], bool):
                raise CustomException(f"Parser '{executable_path}' returned JSON dictionary with unexpected type for value of key 'success', which is '{type(processed_dict['success']).__name__}' instead of 'bool'.")

            if not isinstance(processed_dict['message'], str):
                raise CustomException(f"Parser '{executable_path}' returned JSON dictionary with unexpected type for value of key 'message', which is '{type(processed_dict['message']).__name__}' instead of 'str'.")

            if not isinstance(processed_dict['completion'], dict):
                raise CustomException(f"Parser '{executable_path}' returned JSON dictionary with unexpected type for value of key 'completion', which is '{type(processed_dict['completion']).__name__}' instead of 'dict'.")

            succeeded = True

        except subprocess.TimeoutExpired:
            error_str = f"Parser '{executable_path}' timed out after '{timeout:.1f}s'."
        except CustomException as e:
            error_str = str(e)

        if succeeded:
            self._logger.debug(f"Parser '{executable_path}' succeeded.")
            if not processed_dict['success']:
                self._logger.error(processed_dict['message'])
        else:
            self._logger.error(error_str)
            processed_dict = response_dict
            processed_dict['success'] = False
            processed_dict['message'] = error_str
            processed_dict['completion']['logs'].append(error_str)

        return processed_dict['success'], processed_dict['message'], processed_dict['completion']

    def post_process_completion(self, request, reasoning, text, tool_calls, is_valid, corrections, usage, logs):
        assert usage is not None
        assert 'duration' in usage

        if not is_valid:
            error_log_i = len(logs) - 2 # last is duration log
            if len(self.messages) > self.message_length_original:
                num_remove = len(self.messages) - self.message_length_original
                if num_remove == 1:
                    logs.append("Removing invalid completion from context.")
                else:
                    logs.append(f"Removing '{num_remove}' invalid messages from context (invalid completions + correction messages).")
            self.messages = self.messages[:self.message_length_original]
            assert len(self.messages) == self.message_length_original, f"Expected context to contain '{self.message_length_original}' message{'s' if self.message_length_original == 1 else 's'} but it contains '{len(self.messages)}'."
        elif corrections > 0:
            logs.append(f"Completion is valid after '{corrections}' correction attempt{'' if corrections == 1 else 's'}.")
            self._logger.info(logs[-1])
            if len(self.messages) > self.message_length_original + 2:
                num_remove = len(self.messages) - self.message_length_original - 2
                if num_remove == 1:
                    logs.append("Removing invalid completion from context.")
                else:
                    logs.append(f"Removing '{num_remove}' invalid messages from context (invalid completions + correction messages).")
            self.messages = self.messages[:self.message_length_original + 1] + [self.messages[-1]]
            assert len(self.messages) == self.message_length_original + 2, f"Expected context to contain '{self.message_length_original + 2}' messages but it contains '{len(self.messages)}'."
        else:
            assert len(self.messages) == self.message_length_original + 2, f"Expected context to contain '{self.message_length_original + 2}' messages but it contains '{len(self.messages)}'."

        self.update_awaited_tool_responses()

        response = CompletionsPrompt.Response()
        response.success = is_valid

        completion = {'usage': usage}
        if reasoning is not None:
            completion['reasoning'] = reasoning
        if len(tool_calls) > 0:
            completion['tools'] = []
        for i, call in enumerate(tool_calls):
            if call['arguments'] == "": # fix empty arguments (e.g. Claude does that)
                logs.append(f"Fixing empty arguments of tool call '{call['name']}' to empty dictionary.")
                self._logger.debug(logs[-1])
                call['arguments'] = r"{}"
            call['arguments'] = json.loads(call['arguments'])
            completion['tools'].append(call)
        if text is not None:
            if request.response_type == "json":
                text = json.loads(text)
            elif self.parameters.normalize_text_completion:
                logs.append("Normalizing text completion.")
                self._logger.debug(logs[-1])
                text = normalize_string(text)
            completion['text'] = text

        if is_valid:
            if 'tokens_output' in usage:
                logs.append(f"Generated completion with '{usage['tokens_output']}' token{'' if usage['tokens_output'] == 1 else 's'} in '{usage['duration']:.1f}s'.")
            else:
                logs.append(f"Generated completion in '{usage['duration']:.1f}s'.")
            completion['logs'] = logs
            response.message = logs[-1]

            for parser in self.parameters.completion_parsers:
                response.success, response.message, completion = self.execute_parser(
                    executable_path=parser,
                    success=response.success,
                    message=response.message,
                    completion=completion,
                    timeout=self.parameters.completion_parsers_timeout,
                )
                if not response.success:
                    break
            else:
                self._logger.info(response.message)
        else:
            completion['logs'] = logs
            response.completion = json.dumps(completion)
            response.message = logs[error_log_i]

        response.completion = json.dumps(completion)

        return response

    # Callbacks

    def prompt(self, request, response):
        self._logger.debug("prompt(): start")

        assert not self.is_prompting
        self.is_prompting = True

        self.update_awaited_tool_responses()

        messages, check_prompt_response = self.check_prompt_validity(request)
        if check_prompt_response is not None:
            response = check_prompt_response
            self.is_prompting = False
            self._logger.debug("prompt(): end")
            return response

        self.add_request_to_context(request, messages)

        if request.response_type == "none":
            response = self.get_no_completion_response()
            self.is_prompting = False
            self._logger.debug("prompt(): end")
            return response

        response = self.generate_completion(request)

        self.is_prompting = False
        self._logger.debug("prompt(): end")
        return response

    def interrupt(self, request, response):
        if self.is_prompting:
            self._logger.info("Interrupting completion...")
            while True:
                try:
                    self.pipe[0].send("EXTERNAL")
                except Exception:
                    pass

                if self.is_prompting:
                    self._logger.debug("Waiting until completion is interrupted...", throttle_duration_sec=1.0, skip_first=True)
                    time.sleep(0.1)
                else:
                    break

            response.success = True
            response.message = "Interrupted completion."
            self._logger.info(response.message)
        else:
            response.success = True
            response.message = "There is no completion in progress."
            self._logger.debug("Ignored attempt to interrupt inactive completion.")

        return response

    def get_tools(self, request, response):
        response.success = True
        if self.tools is None:
            response.message = "There are no tools defined."
            response.tools = []
        else:
            response.tools = [json.dumps(tool['function']) for tool in self.tools]
            response.message = "Retrieved tools."
        self._logger.debug(response.message)
        return response

    def set_tools(self, request, response):
        self._logger.debug("Receiving tool update")

        response.success = True

        if len(request.tools) == 0:
            if self.tools is None:
                response.message = "Kept zero tools defined."
                self._logger.debug(response.message)
            else:
                self.tools = None
                response.message = "Undefined all tools."
                self._logger.info(response.message)

        else:
            used_names = []
            tools = []

            for i in range(len(request.tools)):
                try:
                    tools.append(json.loads(request.tools[i]))
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to parse function '{request.tools[i]}' as JSON: {repr(e)}"
                    break

                keys_required = {'name', 'description'} # OpenAI allows omitting 'parameters', Mistral does not, and OpenRouter does with some models.
                keys_optional = {'strict', 'parameters'}

                if not (set(tools[-1].keys()).issubset(keys_required | keys_optional) and keys_required.issubset(tools[-1])):
                    response.success = False
                    response.message = f"Function '{i}' does not satisfy the required format: The top level keys must be {list(keys_required)} and optionally {list(keys_optional)} instead of {list(tools[-1].keys())}."
                    break

                if not isinstance(tools[-1]['name'], str):
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format: The field 'name' must be of type 'str' instead of '{type(tools[-1]['name']).__name__}'."
                    break

                if tools[-1]['name'] in used_names:
                    response.success = False
                    response.message = f"All functions must feature a unique name - The name '{tools[-1]['name']}' is featured more than once."
                    break

                used_names.append(tools[-1]['name'])

                if not isinstance(tools[-1]['description'], str):
                    response.success = False
                    response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format: The field 'description' must be of type 'str' instead of '{type(tools[-1]['description']).__name__}'."
                    break

                if 'strict' in tools[-1]:
                    if not isinstance(tools[-1]['strict'], bool):
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format: The field 'strict' must be of type 'bool' instead of '{type(tools[-1]['strict']).__name__}'."
                        break

                if 'parameters' in tools[-1]:
                    if not isinstance(tools[-1]['parameters'], dict):
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format: The field 'parameters' must be of type 'dict' instead of '{type(tools[-1]['parameters']).__name__}'."
                        break

                    keys_required = {'type', 'properties'}
                    keys_optional = {'required', 'additionalProperties'}
                    if not set(tools[-1]['parameters'].keys()).issubset(keys_required | keys_optional) and keys_required.issubset(tools[-1]['parameters']):
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format: The field 'parameters' must contain the keys {list(keys_required)} and optionally {list(keys_optional)}."
                        break

                    if tools[-1]['parameters']['type'] != "object":
                        response.success = False
                        response.message = f"The function '{tools[-1]['name']}' does not satisfy the required format: The field 'parameters'::'type' must be set to 'object'."
                        break

                    success, message = self.validate_tool_properties(tools[-1]['parameters'], tools[-1]['name'])
                    if not success:
                        response.success = False
                        response.message = message
                        break

            else:
                for i, f in enumerate(tools):
                    tools[i] = {'type': "function", 'function': f}

                if tools == self.tools:
                    response.message = "Kept all tool definitions."
                    self._logger.debug(response.message)
                else:
                    if self.tools is None:
                        response.message = "Set tools."
                        tool_msg = "Defined tools:\n\n" + str('\n\n'.join([f"{i}: {json.dumps(tool['function'], indent=2)}" for i, tool in enumerate(tools)])) + "\n"
                    else:
                        response.message = "Updated tools."

                        updates = 0
                        for tool in tools:
                            if tool in self.tools:
                                updates += 1
                        tool_msg = "Updated tools:\n\n" + str('\n\n'.join([f"{i}{'*' if tool in self.tools else ''}: {json.dumps(tool['function'], indent=2)}" for i, tool in enumerate(tools)])) + "\n"
                        if updates > 0:
                            tool_msg = f"{tool_msg}\n*tool existed before ({updates})\n"

                    log_lines(tool_msg, line_length=self.parameters.log_line_length, line_highlight="| ", block_format=False, allow_empty_lines=True, logger=self._logger, severity=20)

                    self.tools = tools

            if not response.success:
                if self.tools is None:
                    self._logger.error(f"Failed to set tools: {response.message}")
                else:
                    self._logger.error(f"Failed to update tools: {response.message}")

        return response

    def get_context(self, request, response):
        response.success = True
        response.message = "Retrieved context."
        messages = copy.deepcopy(self.messages)
        response.context = [json.dumps(m) for m in messages]
        response.messages = len(self.messages)
        self._logger.debug(response.message)

        return response

    def set_context(self, request, response):
        self._logger.debug("Receiving context update")

        debug_success = False

        if request.mode == "reset":
            if len(request.new_messages) > 0:
                messages_before = copy.deepcopy(self.messages)
                new_messages = []
                for message in request.new_messages:
                    try:
                        message = json.loads(message)
                    except Exception as e:
                        response.success = False
                        response.message = f"Failed to encode message '{message}' (index '{len(self.messages)}') as JSON: {e}"
                        self._logger.error(response.message)
                        return response
                    else:
                        new_messages.append(message)

            if len(self.messages) == 0:
                response.message = "Kept empty context."
                debug_success = True
            else:
                self.messages = []
                self.update_awaited_tool_responses()
                response.message = "Cleared context."
            if len(request.new_messages) == 0:
                response.success = True
            else:
                for message in new_messages:
                    try:
                        self.check_message_validity(message)
                        message = self.encode_files(message)
                    except CustomException as e:
                        response.success = False
                        response.message = f"Failed to build context at message '{message}' (index '{len(self.messages)}'): {e}"
                        self.messages = messages_before
                        self.update_awaited_tool_responses()
                        self._logger.error(response.message)
                        return response
                    else:
                        self.messages.append(message)
                        self.update_awaited_tool_responses()
                response.success = True
                response.message = f"Set new context with '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."
                debug_success = False

        elif request.mode == "insert":
            if len(request.new_messages) == 0:
                response.success = False
                response.message = "Cannot insert messages into context without providing one."
                self._logger.error(response.message)
                return response
            if request.indexing_last_to_first:
                if len(self.messages) == 0:
                    i = len(self.messages) - request.index
                else:
                    i = len(self.messages) - 1 - request.index
            else:
                i = request.index

            new_messages = []
            for message in request.new_messages:
                try:
                    message = json.loads(message)
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to encode message '{message}' (index '{len(self.messages)}') as JSON: {e}"
                    self._logger.error(response.message)
                    return response
                else:
                    new_messages.append(message)

            messages_before = copy.deepcopy(self.messages)

            self.messages = []
            self.update_awaited_tool_responses()
            for message in messages_before[:i] + new_messages + messages_before[i:]:
                try:
                    self.check_message_validity(message)
                    message = self.encode_files(message)
                except CustomException as e:
                    response.success = False
                    response.message = f"Failed to build context at message '{message}' (index '{len(self.messages)}'): {e}"
                    self.messages = messages_before
                    self.update_awaited_tool_responses()
                    self._logger.error(response.message)
                    return response
                else:
                    self.messages.append(message)
                    self.update_awaited_tool_responses()
            response.success = True
            response.message = f"Inserted '{len(new_messages)}' message{'' if len(new_messages) == 1 else 's'} into context."

        elif request.mode == "replace":
            if len(request.new_messages) == 0:
                response.success = False
                response.message = "Cannot replace message in context without providing one."
                self._logger.error(response.message)
                return response
            if request.indexing_last_to_first:
                if len(self.messages) == 0:
                    i = len(self.messages) - request.index
                else:
                    i = len(self.messages) - 1 - request.index
            else:
                i = request.index
            if i < 0 or i > len(self.messages) - 1:
                response.success = False
                response.message = f"Cannot replace message at index '{i}' in context containing '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."
                self._logger.error(response.message)
                return response

            new_messages = []
            for message in request.new_messages:
                try:
                    message = json.loads(message)
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to encode message '{message}' (index '{len(self.messages)}') as JSON: {e}"
                    self._logger.error(response.message)
                    return response
                else:
                    new_messages.append(message)

            messages_before = copy.deepcopy(self.messages)

            self.messages = []
            self.update_awaited_tool_responses()
            for message in messages_before[:i] + new_messages + messages_before[i + len(new_messages):]:
                try:
                    self.check_message_validity(message)
                    message = self.encode_files(message)
                except CustomException as e:
                    response.success = False
                    response.message = f"Failed to build context at message '{message}' (index '{len(self.messages)}'): {e}"
                    self.messages = messages_before
                    self.update_awaited_tool_responses()
                    self._logger.error(response.message)
                    return response
                else:
                    self.messages.append(message)
                    self.update_awaited_tool_responses()
            response.success = True
            added = len(self.messages) - len(messages_before)
            if added == 0:
                response.message = f"Replaced '{len(new_messages)}' message{'' if len(new_messages) == 1 else 's'} in context."
            else:
                response.message = f"Replaced '{len(new_messages) - added}' and added '{added}' message{'' if added == 1 else 's'} to context."

        elif request.mode == "remove":
            if request.indexing_last_to_first:
                if len(self.messages) == 0:
                    i = len(self.messages) - request.index
                else:
                    i = len(self.messages) - 1 - request.index
            else:
                i = request.index
            if i < 0 or i > len(self.messages) - 1:
                response.success = False
                response.message = f"Cannot remove message '{i}' from context containing '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."
                self._logger.error(response.message)
                return response

            messages_before = copy.deepcopy(self.messages)

            self.messages = []
            self.update_awaited_tool_responses()
            for j, message in enumerate(copy.deepcopy(messages_before)):
                if i == j:
                    continue
                try:
                    self.check_message_validity(message)
                    message = self.encode_files(message)
                except CustomException as e:
                    response.success = False
                    response.message = f"Failed to build context at message '{message}' (index '{len(self.messages)}'): {e}"
                    self.messages = messages_before
                    self.update_awaited_tool_responses()
                    self._logger.error(response.message)
                    return response
                else:
                    self.messages.append(message)
                    self.update_awaited_tool_responses()
            response.success = True
            response.message = f"Removed message '{i}' from context."

        else:
            valid_modes = ["reset", "insert", "replace", "remove"]
            response.success = False
            response.message = f"Cannot set context with invalid mode '{request.mode}'. Valid modes: {valid_modes}"
            self._logger.error(response.message)
            return response

        if debug_success:
            self._logger.debug(response.message)
        else:
            self._logger.info(response.message)

        return response

    def reset_parameters(self, request, response):
        response.success = True
        response.message = ""

        self.api_endpoints = self.api_endpoints_default
        successes, messages = self.parameter_handler.update_dict(self.parameter_defaults)

        if all(successes):
            response.message = "Reset parameters to default values."
            self._logger.debug(response.message)
        else:
            failed_messages = [messages[i] for i in range(len(messages)) if not successes[i]]
            response.message = f"Failed to reset parameters to default values: {failed_messages}"
            self._logger.warn("Failed to reset parameters to default values.")

        return response

    def publish_status(self):
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK # OK, WARN, ERROR, STALE
        status.name = self.node_name
        status.message = "status"
        status.hardware_id = "completions"

        kv = KeyValue()
        kv.key = "stamp"
        now = self.get_clock().now().seconds_nanoseconds()
        kv.value = f"{now[0]}.{now[1]}"
        status.values.append(kv)

        kv = KeyValue()
        kv.key = "is prompting"
        kv.value = f"{self.is_prompting}"
        status.values.append(kv)

        if self.tools is None:
            function_count = 0
        else:
            function_count = len(self.tools)

        kv = KeyValue()
        kv.key = "tool count"
        kv.value = str(function_count)
        status.values.append(kv)

        for i in range(function_count):
            kv = KeyValue()
            kv.key = "tool " + str(i)
            kv.value = str(self.tools[i])
            status.values.append(kv)

        kv = KeyValue()
        kv.key = "awaited tool responses"
        kv.value = str(self.awaited_tool_responses)
        status.values.append(kv)

        messages = copy.deepcopy(self.messages)

        message_count = len(messages)

        kv = KeyValue()
        kv.key = "message count"
        kv.value = str(message_count)
        status.values.append(kv)

        for i in range(message_count):
            kv = KeyValue()
            kv.key = "message " + str(i)
            kv.value = str(messages[i])[:self.parameters.log_line_length] # TODO do proper formatting as in logs (cache them there, use them here)
            status.values.append(kv)

        self.pub_status.publish(status)

def main(args=None):
    start_and_spin_node(Completions, args=args)

if __name__ == '__main__':
    main()
