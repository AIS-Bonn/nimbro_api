#!/usr/bin/env python3

import os
import copy
import json
import time

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange
from ament_index_python.packages import get_package_prefix

from nimbro_api.utils.misc import count_tokens
from nimbro_api.utils.node import start_and_spin_node
from nimbro_api.utils.misc import read_json, write_json
from nimbro_api.utils.parameter_handler import ParameterHandler
from nimbro_api_interfaces.srv import GetEmbeddings
from nimbro_api_interfaces.msg import Embedding, ApiUsage

### <Parameter Defaults>

node_name = "embeddings"
logger_level = 10

probe_api_connection = True
api_endpoint = "OpenAI"
model_name = "text-embedding-3-large" # "text-embedding-3-large", # "mistral-embed", # "ais/embedding-llm"

cache_use = True
cache_read_once = True
# cache_write_lazy = True
# cache_write_interval = 30.0
cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "embeddings")
cache_file = "cache_embeddings_index.json"

monitor_usage = True

## non-params

embeddings_per_file = 100
embeddings_name_template = "cache_embeddings_{file_id}.json"
max_texts_per_post = 100 # batches posts including more texts to mistral to stay below token limit

api_endpoints = {
    'OpenAI': {
        'api_flavor': "openai",
        'models_url': "https://api.openai.com/v1/models",
        'embeddings_url': "https://api.openai.com/v1/embeddings",
        'key_type': "environment",
        'key_value': "OPENAI_API_KEY"
    },
    'Mistral AI': {
        'api_flavor': "mistral",
        'models_url': "https://api.mistral.ai/v1/models",
        'embeddings_url': "https://api.mistral.ai/v1/embeddings",
        'key_type': "environment",
        'key_value': "MISTRAL_API_KEY"
    },
    'vLLM': {
        'api_flavor': "openai",
        'models_url': "http://localhost:8000/v1/models",
        'embeddings_url': "http://localhost:8000/v1/embeddings",
        'key_type': "environment",
        'key_value': "VLLM_API_KEY"
    }
}

### </Parameter Defaults>

class Embeddings(Node):

    def __init__(self):
        super().__init__(node_name)
        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self.endpoint_keys_required = {'name', 'api_flavor', 'embeddings_url', 'key_type', 'key_value'}
        self.endpoint_keys_optional = {'models_url'}
        self.endpoint_key_type_values = ["environment", "plain"]
        self.endpoint_api_flavor_values = ["openai", "mistral"]

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
        descriptor.description = f"Sets the API endpoint defining API flavor, Models & Embeddings URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}."
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
        descriptor.name = "cache_use"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Attempt to retrieve embeddings from cached results."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_use, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_read_once"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Read embeddings cache file once when required and keep it in memory instead of loading it every time."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_read_once, descriptor)

        # descriptor = ParameterDescriptor()
        # descriptor.name = "cache_write_lazy"
        # descriptor.type = ParameterType.PARAMETER_BOOL
        # descriptor.description = "Write embeddings cache file in fixed intervals instead of writing it with every update."
        # descriptor.read_only = False
        # self.parameter_descriptors.append(descriptor)
        # self.declare_parameter(descriptor.name, cache_write_lazy, descriptor)

        # descriptor = ParameterDescriptor()
        # descriptor.name = "cache_write_interval"
        # descriptor.type = ParameterType.PARAMETER_DOUBLE
        # descriptor.description = "Minimum time in seconds in which the embeddings cache file is written if cache_write_lazy is active."
        # descriptor.read_only = True
        # float_range = FloatingPointRange()
        # float_range.from_value = 10.0
        # float_range.to_value = 3600.0
        # float_range.step = 0.0
        # descriptor.floating_point_range.append(float_range)
        # self.parameter_descriptors.append(descriptor)
        # self.declare_parameter(descriptor.name, cache_write_interval, descriptor)

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

        descriptor = ParameterDescriptor()
        descriptor.name = "monitor_usage"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Tokenize input strings to monitor usage."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, monitor_usage, descriptor)

        self.parameter_handler.all_declared()

        qos_profile_srv = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)
        self.srv_embeddings = self.create_service(GetEmbeddings, f"{self.node_namespace}/{self.node_name}/get_embeddings".replace("//", "/"), self.get_embeddings_callack, qos_profile=qos_profile_srv, callback_group=ReentrantCallbackGroup())

        qos_profile_pub = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_ALL, depth=10)
        self.pub_usage = self.create_publisher(ApiUsage, f"{self.node_namespace}/api_usage".replace("//", "/"), qos_profile=qos_profile_pub, callback_group=MutuallyExclusiveCallbackGroup())

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

        elif parameter.name == "cache_use":
            self.cache_use = parameter.value
            self.index = None

        elif parameter.name == "cache_read_once":
            self.cache_read_once = parameter.value
            if not self.cache_read_once:
                self.index = None

        # elif parameter.name == "cache_write_lazy":
        #     self.cache_write_lazy = parameter.value

        # elif parameter.name == "cache_write_interval":
        #     self.cache_write_interval = parameter.value

        elif parameter.name == "cache_folder":
            if parameter.value == "":
                self.cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache")
                self._node.get_logger().info(f"Interpreting empty parameter 'cache_folder' as '{self.cache_folder}'")
            else:
                self.cache_folder = parameter.value

        elif parameter.name == "cache_file":
            self.cache_file = parameter.value

        elif parameter.name == "monitor_usage":
            self.monitor_usage = parameter.value

        else:
            return None, None

        return success, reason

    def validate_api_endpoint(self, api_endpoint):
        try:
            json_object = json.loads(api_endpoint)
        except Exception:
            success = False
            message = f"Value must be a name of an existing endpoints in {list(self.api_endpoints.keys())} or a valid JSON encoded dictionary containing a new endpoint."
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

    # Embeddings Pipeline

    def embeddings_post(self, texts, model, api_url, api_key):
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }

        data = {
            'input': texts,
            'model': model,
            'encoding_format': "float"
        }

        # TODO add timeout
        # try:
        #     requests.post(url, data=payload, timeout=5)
        # except requests.Timeout:
        #     # back off and retry
        #     pass
        # except requests.ConnectionError:
        #     pass

        self.get_logger().debug(f"Posting request: {data}\n to '{api_url}'")
        tic = time.perf_counter()
        try:
            response = requests.post(api_url, headers=headers, json=data, stream=False)
        except Exception as e:
            toc = time.perf_counter()
            self.get_logger().debug(f"Error occured after '{toc - tic:.3f}s': {e}")
            success = False
            message = f"{e}"
            embeddings = None
        else:
            toc = time.perf_counter()
            self.get_logger().debug(f"Received response after '{toc - tic:.3f}s'")

            if response.status_code == 200:
                response = response.json()
                success = True
                message = f"Successfully retrieved '{len(response['data'])}' embedding{'' if len(response['data']) == 1 else 's'}."
                embeddings = [response['data'][i]['embedding'] for i in range(len(response['data']))]
            else:
                success = False
                message = f"HTTP-Error: {response.text}"
                embeddings = None

        return success, message, embeddings

    def save_usage(self, texts):
        if self.monitor_usage:
            num_tokens = 0
            tic = time.perf_counter()
            for text in texts:
                try:
                    num_tokens += count_tokens(text, "cl100k_base") # for third-generation embedding
                except ModuleNotFoundError:
                    self.get_logger().warn("Cannot monitor usage because the module tiktoken is not installed")
                    self.monitor_usage = False # TODO properly deactivate using "from rcl_interfaces.srv import SetParameters"
                    return
            self.get_logger().debug(f"Tokenizing uncached text{'' if len(texts) == 1 else 's'} took '{time.perf_counter() - tic:.3f}s'")
        else:
            return

        usage = ApiUsage()
        usage.api_type = "embeddings"
        usage.api_endpoint = self.api_endpoint
        usage.model_name = self.model_name
        usage.input_tokens_uncached = num_tokens
        usage.input_tokens_cached = 0
        usage.output_tokens = 0

        self.pub_usage.publish(usage)

        self.get_logger().debug(f"Retrieving missing embedding{'' if len(texts) == 1 else 's'} consumed '{num_tokens}' token{'' if num_tokens == 1 else 's'}")

    def get_embeddings(self, texts):
        # parse argument

        if len(texts) == 0:
            return True, "Successfully retrieved '0' embeddings.", []
        if not all(isinstance(t, str) for t in texts):
            message = "All items in list 'texts' must be of type 'str'."
            self.get_logger().error(message[:-1])
            return False, message, None
        text_formatted = [t.replace("\n", " ") for t in texts]

        for t in text_formatted:
            if t == "":
                message = "None of the passed texts must be empty."
                self.get_logger().error(message[:-1])
                return False, message, None

        embeddings = [None] * len(text_formatted)

        # read cache

        cache_use = copy.copy(self.cache_use)

        if not cache_use:
            self.index = None
            missing_idx = list(range(len(text_formatted)))
        else:
            if not self.cache_read_once or self.index is None:
                # read index file
                index_path = os.path.join(self.cache_folder, self.cache_file)
                if os.path.exists(index_path):
                    success, _, self.index = read_json(file_path=index_path, logger=self.get_logger())
                    if success:
                        if not isinstance(self.index, dict):
                            success = False
                            self.get_logger().error(f"Expected content of index file to be of type 'dict', but it is of type '{type(self.index).__name__}'", throttle_duration_sec=10.0)
                        elif 'files' not in self.index or 'texts' not in self.index:
                            success = False
                            self.get_logger().error("Expected index file to feature the keys 'files' and 'texts'", throttle_duration_sec=10.0)
                        elif not isinstance(self.index['files'], list) or not isinstance(self.index['texts'], dict):
                            success = False
                            self.get_logger().error("Expected index file keys 'files' and 'texts' to feature values of type 'list' and 'dict'", throttle_duration_sec=10.0)
                    if not success:
                        self.get_logger().warn("Initializing new index file. Corrupt cache files might get overwritten!", throttle_duration_sec=10.0)
                        self.index = {'files': [], 'texts': {}}
                else:
                    success = False
                    self.get_logger().info("Initializing new index file")
                    self.index = {'files': [], 'texts': {}}

                self.embeddings_files = {}

            # assess embeddings listed in index

            corrupted = False

            if self.index['texts'].get(self.model_name) is None:
                self.get_logger().debug(f"Cannot find model '{self.model_name}' in index file")
                missing_idx = list(range(len(text_formatted)))
            else:
                missing_idx = []
                missing_index_tuples = []
                missing_file_idx = []
                for i, t in enumerate(text_formatted):
                    if self.index['texts'][self.model_name].get(t) is None:
                        self.get_logger().debug(f"Cannot find text '{t}' in index file")
                        missing_idx.append(i)
                    else:
                        if not isinstance(self.index['texts'][self.model_name][t], list):
                            self.get_logger().error(f"Cannot find text '{t}' in index file: File is corrupted (value of text is not a list)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        elif not len(self.index['texts'][self.model_name][t]) == 2:
                            self.get_logger().error(f"Cannot find text '{t}' in index file: File is corrupted (value of text is not of length 2)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        elif not isinstance(self.index['texts'][self.model_name][t][0], int) or not isinstance(self.index['texts'][self.model_name][t][1], int):
                            self.get_logger().error(f"Cannot find text '{t}' in index file: File is corrupted (values of text is not a list of integers)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        elif self.index['texts'][self.model_name][t][0] >= len(self.index['files']):
                            self.get_logger().error(f"Cannot find text '{t}' in index file: File is corrupted (value of text points to file that does not exist)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        else:
                            file_id = self.index['texts'][self.model_name][t][0]
                            embedding_id = self.index['texts'][self.model_name][t][1]
                            self.get_logger().debug(f"Text '{t}' found in index file (file: '{file_id}', element: '{embedding_id}')")
                            missing_index_tuples.append((file_id, embedding_id, i))
                            missing_file_idx.append(file_id)

                # read embeddings files

                missing_file_idx = list(set(missing_file_idx))

                for i in missing_file_idx:
                    if i not in self.embeddings_files:
                        if not isinstance(self.index['files'][i], list):
                            self.get_logger().error(f"Cannot find embeddings file '{i}' in index file: File is corrupted (value of file is not a list)", throttle_duration_sec=10.0)
                            corrupted = True
                        elif not len(self.index['files'][i]) == 2:
                            self.get_logger().error(f"Cannot find embeddings file '{i}' in index file: File is corrupted (value of file is not of length 2)", throttle_duration_sec=10.0)
                            corrupted = True
                        elif not isinstance(self.index['files'][i][0], str):
                            self.get_logger().error(f"Cannot find embeddings file '{i}' in index file: File is corrupted (value of first element is not a string)", throttle_duration_sec=10.0)
                            corrupted = True
                        else:
                            embeddings_file_path = os.path.join(self.cache_folder, self.index['files'][i][0])
                            success, _, embeddings_file = read_json(file_path=embeddings_file_path, logger=self.get_logger())
                            if not success:
                                pass
                            elif not isinstance(embeddings_file, list):
                                self.get_logger().warn(f"Embeddings file '{embeddings_file_path}' is corrupted (content is not a list) (1)", throttle_duration_sec=10.0)
                                corrupted = True
                            elif len(embeddings_file) != self.index['files'][i][1]:
                                self.get_logger().warn(f"Embeddings file '{embeddings_file_path}' is corrupted (actual size '{len(embeddings_file)}' does not match size in index '{self.index['files'][i][1]}') (1)", throttle_duration_sec=10.0)
                                corrupted = True
                            else:
                                self.embeddings_files[i] = embeddings_file

                # collect embeddings

                for file_id, embedding_id, missing_id in missing_index_tuples:
                    if file_id in self.embeddings_files:
                        if embedding_id < len(self.embeddings_files[file_id]):
                            self.get_logger().debug(f"Found cached embedding for text '{text_formatted[missing_id]}' (file: '{file_id}', element: '{embedding_id}')")
                            embeddings[missing_id] = self.embeddings_files[file_id][embedding_id]
                        else:
                            self.get_logger().error(f"Cannot find embedding '{missing_id}' in embeddings file '{file_id}': File is corrupted (file only contains '{len(self.embeddings_files[file_id])}' embeddings)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(missing_id)
                    else:
                        missing_idx.append(missing_id)

        if len(missing_idx) == 0:
            self.get_logger().debug("All requested embeddings were found in cache")
        else:
            # retrieve API key

            if self.api_endpoints[self.api_endpoint]['key_type'] == "environment":
                api_key = os.getenv(self.api_endpoints[self.api_endpoint]['key_value'])
                if api_key is None:
                    message = f"Failed to read API key from environment variable '{self.api_endpoints[self.api_endpoint]['key_value']}')."
                    self.get_logger().error(message)
                    return False, message, None
            else:
                api_key = self.api_endpoints[self.api_endpoint]['key_value']
            self.get_logger().debug(f"Retrieved API key '{api_key}'")

            # retrieve missing embeddings

            self.get_logger().info(f"Retrieving '{len(missing_idx)}' missing embedding{'' if len(missing_idx) == 1 else 's'} from API")
            missing_texts = [text_formatted[i] for i in missing_idx]

            if self.api_endpoints[self.api_endpoint]['api_flavor'] == "mistral": # mistral complains about exceeding maximum input tokens
                floor_mod = (len(missing_texts) // max_texts_per_post, len(missing_texts) % max_texts_per_post)
                missing_embeddings = []
                for i in range(floor_mod[0]):
                    success, message, embeddings_batch = self.embeddings_post(
                        texts=missing_texts[i * max_texts_per_post: (i + 1) * max_texts_per_post],
                        model=self.model_name,
                        api_url=self.api_endpoints[self.api_endpoint]['embeddings_url'],
                        api_key=api_key
                    )
                    if success:
                        missing_embeddings += embeddings_batch
                    else:
                        self.get_logger().error(message)
                        return False, message, None
                if floor_mod[1] > 0:
                    success, message, embeddings_batch = self.embeddings_post(
                        texts=missing_texts[floor_mod[0] * max_texts_per_post:],
                        model=self.model_name,
                        api_url=self.api_endpoints[self.api_endpoint]['embeddings_url'],
                        api_key=api_key
                    )
                    if success:
                        missing_embeddings += embeddings_batch
                    else:
                        self.get_logger().error(message)
                        return False, message, None
            else:
                success, message, missing_embeddings = self.embeddings_post(
                    texts=missing_texts,
                    model=self.model_name,
                    api_url=self.api_endpoints[self.api_endpoint]['embeddings_url'],
                    api_key=api_key
                )
            if success:
                self.get_logger().debug(f"Retrieved '{len(missing_idx)}' missing embedding{'' if len(missing_idx) == 1 else 's'} from API")
                self.save_usage(missing_texts)
            else:
                self.get_logger().error(message)
                return False, message, None

            # fill up missing embeddings

            for i, j in enumerate(missing_idx):
                embeddings[j] = missing_embeddings[i]

            # add missing_embeddings to cache and embeddings_files, and overwrite touched files

            if cache_use and len(missing_idx) > 0 and self.index is not None and not corrupted:
                touched_embeddings_files = []
                for i in missing_idx:
                    for j, (file_name, file_size) in enumerate(self.index['files']):
                        if file_size < embeddings_per_file:
                            if j not in self.embeddings_files and j:
                                embeddings_file_path = os.path.join(self.cache_folder, self.index['files'][j][0])
                                success, _, embeddings_file = read_json(file_path=embeddings_file_path, logger=self.get_logger())
                                if not success:
                                    pass
                                elif not isinstance(embeddings_file, list):
                                    self.get_logger().warn(f"Embeddings file '{embeddings_file_path}' is corrupted (content is not a list) (2)", throttle_duration_sec=10.0)
                                    self.cache_use = False
                                    corrupted = True
                                    break
                                elif len(embeddings_file) != self.index['files'][j][1]:
                                    self.get_logger().warn(f"Embeddings file '{embeddings_file_path}' is corrupted (actual size '{len(embeddings_file)}' does not match size in index '{self.index['files'][i][1]}') (2)", throttle_duration_sec=10.0)
                                    self.cache_use = False
                                    corrupted = True
                                    break
                                else:
                                    self.embeddings_files[j] = embeddings_file

                            if j in self.embeddings_files:
                                self.index['files'][j][1] += 1
                                self.embeddings_files[j].append(embeddings[i])
                                if j not in touched_embeddings_files:
                                    touched_embeddings_files.append(j)
                                if self.model_name not in self.index['texts']:
                                    self.index['texts'][self.model_name] = {}
                                embedding_id = len(self.embeddings_files[j]) - 1
                                self.index['texts'][self.model_name][text_formatted[i]] = [j, embedding_id]
                                self.get_logger().debug(f"Caching text '{text_formatted[i]}' (file: '{j}', element: '{embedding_id}')")
                                break

                        if corrupted:
                            break

                    else:
                        j = len(self.index['files'])
                        new_name = embeddings_name_template.format(file_id=j)
                        self.get_logger().debug(f"Creating new embeddings file '{new_name}'")
                        self.embeddings_files[j] = [embeddings[i]]
                        if j not in touched_embeddings_files:
                            touched_embeddings_files.append(j)
                        self.index['files'].append([new_name, 1])
                        if self.model_name not in self.index['texts']:
                            self.index['texts'][self.model_name] = {}
                        embedding_id = len(self.embeddings_files[j]) - 1
                        self.index['texts'][self.model_name][text_formatted[i]] = [j, embedding_id]
                        self.get_logger().debug(f"Caching text '{text_formatted[i]}' (file: '{j}', element: '{embedding_id}')")

                    if corrupted:
                        break

                # TODO implement cache_write_lazy
                if not corrupted:
                    if len(touched_embeddings_files) > 0:
                        cache_path = os.path.join(self.cache_folder, self.cache_file)
                        write_json(cache_path, self.index, indent=True, logger=self.get_logger())
                    for i in touched_embeddings_files:
                        embeddings_file_path = os.path.join(self.cache_folder, self.index['files'][i][0])
                        write_json(embeddings_file_path, self.embeddings_files[i], indent=False, logger=self.get_logger())

        # forward results

        if len(str(text_formatted)) > 100:
            self.get_logger().info(f"Successfully retrieved '{len(embeddings)}' embedding{'' if len(embeddings) == 1 else 's'} for text{'' if len(text_formatted) == 1 else 's'}: {str(text_formatted)[:100]}...")
        else:
            self.get_logger().info(f"Successfully retrieved '{len(embeddings)}' embedding{'' if len(embeddings) == 1 else 's'} for text{'' if len(text_formatted) == 1 else 's'}: {text_formatted}")

        return True, f"Successfully retrieved '{len(embeddings)}' embedding{'' if len(embeddings) == 1 else 's'}.", embeddings

    # Callbacks

    def get_embeddings_callack(self, request, response):
        self.get_logger().debug("get_embeddings_callack(): start")

        response.success, response.message, embeddings = self.get_embeddings(texts=request.texts)
        if response.success:
            for embedding_np in embeddings:
                embedding_msg = Embedding()
                embedding_msg.embedding = embedding_np
                response.embeddings.append(embedding_msg)

        self.get_logger().debug("get_embeddings_callack(): end")
        return response

def main(args=None):
    start_and_spin_node(Embeddings, args=args)

if __name__ == '__main__':
    main()
