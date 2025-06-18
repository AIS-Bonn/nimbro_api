#!/usr/bin/env python3

import os
import re
import copy
import json
import time
import threading
import traceback

try:
    import pybase64 as base64
except ImportError:
    import base64
import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange

from nimbro_api_interfaces.srv import GetNimbroVision
from nimbro_api.utils.node import start_and_spin_node
from nimbro_api.utils.parameter_handler import ParameterHandler

### <Parameter Defaults>

node_name = "nimbro_vision"
logger_level = 10

probe_api_connection = True
probe_model_state = True
api_endpoint = "localhost"

## non-params

api_endpoints = {
    'localhost': {
        'mmgroundingdino_url': "http://localhost:9000",
        'mmgroundingdino_key_type': "environment",
        'mmgroundingdino_key_value': "NIMBRO_VISION_API_KEY",

        'sam2_realtime_url': "http://localhost:9001",
        'sam2_realtime_key_type': "environment",
        'sam2_realtime_key_value': "NIMBRO_VISION_API_KEY",

        'dam_url': "http://localhost:9002",
        'dam_key_type': "environment",
        'dam_key_value': "NIMBRO_VISION_API_KEY",

        'kosmos2_url': "http://localhost:9003",
        'kosmos2_key_type': "environment",
        'kosmos2_key_value': "NIMBRO_VISION_API_KEY",

        'florence2_url': "http://localhost:9004",
        'florence2_key_type': "environment",
        'florence2_key_value': "NIMBRO_VISION_API_KEY"
    }
}

### </Parameter Defaults>

class NimbroVision(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self.model_names = ["mmgroundingdino", "sam2_realtime", "dam", "kosmos2", "florence2"]
        self.endpoint_required_sets = [{f"{model}_url", f"{model}_key_type", f"{model}_key_value"} for model in self.model_names]
        self.endpoint_key_type_values = ["environment", "plain"]
        model_names_pattern = "|".join(re.escape(name) for name in self.model_names)
        self.re_pattern_any = re.compile(
            rf"^(?P<model>{model_names_pattern})(?:_(?P<idx>\d+))?_(?P<field>url|key_type|key_value)$"
        )

        for endpoint_name, endpoint in api_endpoints.items():
            assert isinstance(endpoint_name, str), \
                f"Endpoint names must be of type 'str' instead of '{type(endpoint_name).__name__}'."
            assert isinstance(endpoint, dict), \
                f"Endpoint '{endpoint_name}' must be of type 'dict' instead of '{type(endpoint).__name__}'."
            assert all(isinstance(key, str) for key in endpoint), \
                f"Endpoint '{endpoint_name}' must contain only keys of type 'str'."

            instances = {}
            for key in endpoint:
                m = self.re_pattern_any.match(key)
                assert m, (
                    f"Unexpected key '{key}' in endpoint '{endpoint_name}'. "
                    f"Keys must match '<model>_<n?>_(url|key_type|key_value)'."
                )
                model = m.group("model")
                assert model in self.model_names, (
                    f"Unknown model '{model}' in key '{key}'. "
                    f"Expected one of {self.model_names}."
                )
                idx_str = m.group("idx")
                if idx_str:
                    idx = int(idx_str)
                    assert idx != 0, (
                        f"Index for model '{model}' in key '{key}' must be >=1, not 0."
                    )
                else:
                    idx = None
                field = m.group("field")
                instances.setdefault((model, idx), set()).add(field)

            all_expected_keys = set()
            for (model, idx), fields in instances.items():
                suffix = "" if idx is None else f"_{idx}"

                missing = {"url", "key_type", "key_value"} - fields
                assert not missing, (
                    f"Instance '{model}{suffix}' in endpoint '{endpoint_name}' "
                    f"is missing fields: {sorted(missing)}."
                )

                expected_keys = {
                    f"{model}{suffix}_url",
                    f"{model}{suffix}_key_type",
                    f"{model}{suffix}_key_value"
                }
                all_expected_keys |= expected_keys

                for full_key in expected_keys:
                    val = endpoint[full_key]
                    assert isinstance(val, str), (
                        f"Value of '{full_key}' must be str, not '{type(val).__name__}'."
                    )
                    if full_key.endswith("_key_type"):
                        assert val in self.endpoint_key_type_values, (
                            f"'{full_key}' must be one of {self.endpoint_key_type_values}, not '{val}'."
                        )

            extras = set(endpoint) - all_expected_keys
            assert not extras, (
                f"Endpoint '{endpoint_name}' contains unexpected keys: {sorted(extras)}. "
                f"Only full sets of '<model>_<n?>_(url|key_type|key_value)' are allowed."
            )

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
        descriptor.description = "Probes the API endpoint to validate the API key and model name."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, probe_api_connection, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "probe_model_state"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Probes the model state before inference and loads the requested model if required."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, probe_api_connection, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "api_endpoint"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = f"Sets the API endpoint defining URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, api_endpoint, descriptor)

        self.parameter_handler.all_declared()

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)

        self.srv_mmgd = self.create_service(GetNimbroVision, f"{self.node_namespace}/{self.node_name}/mmgroundingdino".replace("//", "/"), self.mmgroundingdino_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_sam2_realtime_update = self.create_service(GetNimbroVision, f"{self.node_namespace}/{self.node_name}/sam2_realtime_update".replace("//", "/"), self.sam2_realtime_update_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_sam2_realtime_track = self.create_service(GetNimbroVision, f"{self.node_namespace}/{self.node_name}/sam2_realtime_track".replace("//", "/"), self.sam2_realtime_track_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_dam = self.create_service(GetNimbroVision, f"{self.node_namespace}/{self.node_name}/dam".replace("//", "/"), self.dam_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_kosmos2 = self.create_service(GetNimbroVision, f"{self.node_namespace}/{self.node_name}/kosmos2".replace("//", "/"), self.kosmos2_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_florence2 = self.create_service(GetNimbroVision, f"{self.node_namespace}/{self.node_name}/florence2".replace("//", "/"), self.florence2_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())

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
                    if self.endpoint_probes.get(self.api_endpoint) is None:
                        success, reason = self.probe_models_api(self.api_endpoint)
                    else:
                        self.get_logger().debug(f"Probing API for endpoint '{self.api_endpoint}' is not required")

        elif parameter.name == "probe_model_state":
            self.probe_model_state = parameter.value

        elif parameter.name == "api_endpoint":
            probe = None
            if parameter.value in list(self.api_endpoints.keys()):
                json_object = None
                if self.endpoint_probes.get(parameter.value) is None:
                    probe = parameter.value
            else:
                success, reason, json_object = self.validate_api_endpoint(parameter.value)
                if success:
                    if json_object['name'] in list(self.api_endpoints.keys()):
                        json_object_without_name = copy.deepcopy(json_object)
                        del json_object_without_name['name']
                        for key in json_object_without_name:
                            if self.api_endpoints[json_object['name']][key] != json_object_without_name[key]:
                                probe = json_object
                    else:
                        probe = json_object

            if success and self.probe_api_connection is True:
                if probe is None:
                    self.get_logger().debug(f"Probing API for endpoint '{parameter.value if json_object is None else json_object['name']}' is not required")
                else:
                    success, reason = self.probe_models_api(probe)

            if success:
                if self.setup_finished:
                    for model in self.model_locks:
                        if self.model_locks[model].locked():
                            self.get_logger().info(f"Waiting for model '{model}' to be released before updating endpoint")
                            self.model_locks[model].acquire()

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

                self.endpoint_models = {}
                self.model_locks = {}
                for endpoint_name in self.api_endpoints:
                    self.endpoint_models[endpoint_name] = []
                    for key in self.api_endpoints[endpoint_name]:
                        if key.endswith("_url"):
                            self.endpoint_models[endpoint_name].append(key[:-4])
                            self.model_locks[key[:-4]] = threading.Lock()
                self.get_logger().debug(f"Models: {self.endpoint_models}")
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
                message = f"JSON encoded endpoint must contain only keys of type 'str' instead of {[type(key).__name__ for key in json_object]}."
            elif 'name' not in json_object:
                success = False
                message = f"JSON encoded endpoint must contain key 'name' but it only contains keys {sorted(json_object.keys())}."
            elif not all(isinstance(json_object[key], str) for key in json_object):
                success = False
                message = f"JSON encoded endpoint must contain only values of type 'str' instead of {[type(json_object[key]).__name__ for key in json_object]}."
            else:
                keys = set(json_object.keys()) - {'name'}
                instances = {}
                for key in keys:
                    match = self.re_pattern_any.match(key)
                    if match is None:
                        success = False
                        message = f"Unexpected key '{key}' not matching pattern '<model>_<n?>_(url|key_type|key_value)'."
                        break

                    model = match.group("model")
                    if model not in self.model_names:
                        success = False
                        message = f"Unknown model '{model}' in key '{key}'. Expected one of {self.model_names}."
                        break

                    idx_str = match.group("idx")
                    if idx_str:
                        idx = int(idx_str)
                        if idx == 0:
                            success = False
                            message = f"Index for model '{model}' in key '{key}' must be >=1, not 0."
                            break
                    else:
                        idx = None
                    field = match.group("field")
                    instances.setdefault(key[:-len(field) - 1], set()).add(field)
                else:
                    all_expected_keys = set()
                    for instance, fields in instances.items():
                        missing = {"url", "key_type", "key_value"} - fields
                        if missing:
                            success = False
                            message = f"Instance '{instance}' is missing fields: {sorted(missing)}."
                            break

                        for field in ["url", "key_type", "key_value"]:
                            full_key = f"{instance}_{field}" if "_" in instance else f"{instance}_{field}"
                            if field == "key_type" and json_object[full_key] not in self.endpoint_key_type_values:
                                success = False
                                message = f"'{full_key}' must be one of {self.endpoint_key_type_values}, not '{json_object[full_key]}'."
                                break
                            all_expected_keys.add(full_key)
                        else:
                            continue
                        break
                    else:
                        extras = keys - all_expected_keys
                        if extras:
                            success = False
                            message = f"JSON encoded endpoint contains unexpected keys: {sorted(extras)}. Only full key sets per model instance are allowed."
                        else:
                            success = True
                            message = ""

        return success, message, json_object

    # API requests

    def probe_models_api(self, api_endpoint):
        if isinstance(api_endpoint, str):
            api_endpoint_name = api_endpoint
            api_endpoint = self.api_endpoints[api_endpoint]
        else:
            api_endpoint_name = api_endpoint['name']

        success = True
        message = ""

        for key in api_endpoint:
            if key[-4:] == "_url":
                model = key[:-4]
            else:
                continue

            if api_endpoint[f"{model}_key_type"] == "environment":
                var_name = api_endpoint[f"{model}_key_value"]
                api_key = os.getenv(var_name)
                if api_key is None:
                    success = False
                    message = f"Error while probing API: Failed to read API key from environment variable '{var_name}')."
                    break
            else:
                api_key = api_endpoint[f"{model}_key_value"]

            if success:
                url = api_endpoint[f"{model}_url"]
                self.get_logger().debug(f"Probing model '{model}' using URL '{url}/model_flavors' and key '{api_key}' of endpoint '{api_endpoint_name}'")
                try:
                    response = requests.get(f"{url}/model_flavors", headers={"Authorization": f"Bearer {api_key}"})
                except Exception as e:
                    success = False
                    message = f"Error while probing API: Failed to get status request: {repr(e)}"
                    break
                else:
                    if response.status_code != 200:
                        success = False
                        message = f"Error while probing API: Request failed with status code '{response.status_code}': {response.text}"
                        break
                    else:
                        available_flavors = response.json().get('flavors', [])
                        if len(available_flavors) == 0:
                            self.get_logger().warn(f"There are no flavors available for model '{model}'")
                        else:
                            self.get_logger().debug(f"Available flavors of model '{model}': {available_flavors}")
                        if api_endpoint_name not in self.endpoint_probes:
                            self.endpoint_probes[api_endpoint_name] = {}
                        self.endpoint_probes[api_endpoint_name][f"{model}_flavors"] = available_flavors
                        self.endpoint_probes[api_endpoint_name]['stamp'] = time.time()

        if success:
            self.get_logger().debug(f"Probes: {self.endpoint_probes}")
        else:
            self.get_logger().error(message)
            if api_endpoint_name in self.endpoint_probes:
                del self.endpoint_probes[api_endpoint_name]

        return success, message

    def handle_request(self, model, request, response):
        self.get_logger().debug(f"handle_request('{model}'): start")
        stamp_start = time.perf_counter()
        response.success = True

        # resolve SAM mode
        sam_track = False
        if model.find("sam2_realtime") > -1:
            if model == "sam2_realtime_track":
                model = "sam2_realtime"
                sam_track = True
            else:
                model = "sam2_realtime"

        api_endpoint = copy.deepcopy(self.api_endpoints[self.api_endpoint])

        # check if model is defnied in endpoint
        if request.model_id > 0:
            model = f"{model}_{request.model_id}"
        if model not in self.endpoint_models[self.api_endpoint]:
            response.success = False
            response.message = f"Model '{model}' is not a known model: {self.endpoint_models[self.api_endpoint]}."

        # check if requested flavor is valid
        if response.success:
            if self.probe_api_connection and not sam_track:
                if request.flavor not in self.endpoint_probes.get(self.api_endpoint, {}).get(f"{model}_flavors", []):
                    response.success = False
                    flavors = self.endpoint_probes[self.api_endpoint][f"{model}_flavors"]
                    response.message = f"Model flavor '{request.flavor}' is not in list of available model flavors {flavors}."

        # check request
        if response.success:
            response.success = False
            try:
                data = json.loads(request.data)
            except Exception as e:
                response.message = f"Failed to parse request field 'data' as JSON: {repr(e)}"
            else:
                response.success = True
                self.get_logger().debug(f"Parsed '{model}' request as JSON")

        # encode images if required
        if response.success:
            response.success = False
            if isinstance(data.get('image'), str):
                if os.path.isfile(data['image']):
                    self.get_logger().debug(f"Encoding image '{data['image']}' into Base64")
                    try:
                        with open(data['image'], "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    except Exception as e:
                        response.message = f"Failed to encode image file '{data['image']}' into Base64: {repr(e)}"
                    else:
                        response.success = True
                        data['image'] = base64_image
                else:
                    response.success = True
                    self.get_logger().debug("Assuming image is Base64 encoded")
            elif isinstance(data.get('images'), list) and all(isinstance(image, str) for image in data['images']):
                for i, image_path in enumerate(data['images']):
                    if os.path.isfile(image_path):
                        self.get_logger().debug(f"Encoding image '{image_path}' into Base64")
                        try:
                            with open(image_path, "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        except Exception as e:
                            response.message = f"Failed to encode image file '{image_path}' into Base64: {repr(e)}"
                            break
                        else:
                            data['images'][i] = base64_image
                    else:
                        self.get_logger().debug(f"Assuming image '{i + 1}' of '{len(data['images'])}' is Base64 encoded")
                else:
                    response.success = True
            else:
                response.success = True

        # retrieve API key
        if response.success:
            response.success = False
            url = api_endpoint[f"{model}_url"]
            if api_endpoint[f"{model}_key_type"] == "environment":
                api_key = os.getenv(api_endpoint[f"{model}_key_value"])
                if api_key is None:
                    var = api_endpoint[f"{model}_key_value"]
                    response.message = f"Failed to read API key from environment variable '{var}')."
                else:
                    response.success = True
            else:
                response.success = True
                api_key = api_endpoint[f"{model}_key_value"]
            if api_key is not None:
                self.get_logger().debug(f"Retrieved '{model}' key '{api_key}'")

        # lock model
        if response.success:
            if self.model_locks[model].locked():
                self.get_logger().info(f"Waiting for model '{model}' to be released before using it")
            self.model_locks[model].acquire()

        # retrieve loaded state/flavor
        if response.success:
            if self.probe_model_state and not sam_track:

                response.success = False
                self.get_logger().debug(f"Requesting '{model}' endpoint status via URL '{url}/status'")
                try:
                    result = requests.get(f"{url}/status", headers={"Authorization": f"Bearer {api_key}"})
                except Exception as e:
                    response.message = f"Failed to get status request: {repr(e)}"
                else:
                    if result.status_code == 200:
                        load = False
                        result = result.json()
                        if model.find(result.get('model_family')) == -1:
                            response.message = f"Provided URL '{url}' hosts wrong model type '{result['model_family']}'."
                        else:
                            response.success = True
                            if result.get('status') is None:
                                load = True
                                self.get_logger().info(f"Provided URL '{url}' currently hosts no model flavor instead of '{request.flavor}'")
                            elif result.get('status', {}).get('flavor') != request.flavor:
                                load = True
                                self.get_logger().info(f"Provided URL '{url}' currently hosts model flavor '{result.get('status', {}).get('flavor')}' instead of '{request.flavor}'")
                            else:
                                self.get_logger().debug(f"Provided URL '{url}' currently hosts the requested model flavor '{request.flavor}'")
                    else:
                        response.message = f"Status request failed with status code '{result.status_code}': {result.text}"

                # load requested model flavor if required
                if response.success and load:
                    response.success = False
                    self.get_logger().debug(f"Requesting '{model}' endpoint via URL '{url}/load' to load model '{request.flavor}'")
                    try:
                        result = requests.post(f"{url}/load", json={'flavor': request.flavor}, headers={"Authorization": f"Bearer {api_key}"})
                    except Exception as e:
                        response.message = f"Failed to POST load request: {repr(e)}"
                    else:
                        if result.status_code == 200:
                            self.get_logger().info(f"Successfully loaded '{model}' model flavor '{request.flavor}': {result.text}")
                            response.success = True
                        else:
                            response.message = f"Load request failed with status code '{result.status_code}': {result.text}"

        # post inference request
        if response.success:
            response.success = False
            if (model == "sam2_realtime" or model == "sam2_realtime_2") and not sam_track:
                suffix = "update"
            else:
                suffix = "infer"
            self.get_logger().debug(f"Requesting endpoint '{model}' inference via URL '{url}/{suffix}'")
            try:
                result = requests.post(f"{url}/{suffix}", json=data, headers={"Authorization": f"Bearer {api_key}"})
            except Exception as e:
                response.message = f"Failed to POST inference request: {repr(e)}"
            else:
                if result.status_code == 200:
                    response.result = result.content.decode('utf-8')
                    response.success = True
                    duration = time.perf_counter() - stamp_start
                    self.get_logger().debug(f"Successfully inferred '{model}' in '{duration:.3f}s': {response.result[:80]}{'...' if len(result.text) >= 80 else ''}")
                    response.message = f"Successfully retrieved '{model}' {'inference' if suffix == 'infer' else 'update'} response after '{duration:.3f}s'."
                else:
                    response.message = f"Model '{model}' {'inference' if suffix == 'infer' else 'update'} failed with status code '{result.status_code}': {result.text}"

        # log
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().error(f"Error occurred while using '{model}': {response.message}")

        # release model
        if model in self.model_locks and self.model_locks[model].locked():
            self.model_locks[model].release()

        self.get_logger().debug(f"handle_request('{model}'): end after '{time.perf_counter() - stamp_start:.3f}s'")
        return response

    # service callbacks

    def mmgroundingdino_callback(self, request, response):
        try:
            response = self.handle_request('mmgroundingdino', request, response)
        except Exception as e:
            self.get_logger().error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def sam2_realtime_update_callback(self, request, response):
        try:
            response = self.handle_request('sam2_realtime_update', request, response)
        except Exception as e:
            self.get_logger().error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def sam2_realtime_track_callback(self, request, response):
        try:
            response = self.handle_request('sam2_realtime_track', request, response)
        except Exception as e:
            self.get_logger().error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def dam_callback(self, request, response):
        try:
            response = self.handle_request('dam', request, response)
        except Exception as e:
            self.get_logger().error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def kosmos2_callback(self, request, response):
        try:
            response = self.handle_request('kosmos2', request, response)
        except Exception as e:
            self.get_logger().error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def florence2_callback(self, request, response):
        try:
            response = self.handle_request('florence2', request, response)
        except Exception as e:
            self.get_logger().error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

def main(args=None):
    start_and_spin_node(NimbroVision, args=args)

if __name__ == '__main__':
    main()
