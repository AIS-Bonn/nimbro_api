#!/usr/bin/env python3

import os
import re
import json
import time
import threading
import traceback

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from nimbro_api_interfaces.srv import NimbroVisionGet
from nimbro_api.misc.common import filter_api_endpoint

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, remove_whitespace, is_base64, is_url, read_as_b64

### <Parameter Defaults>

node_name = "nimbro_vision"
severity = 10

probe_api_connection = True
probe_model_state = True
api_endpoint = "localhost"

## non-params

line_length = 150

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
    },
    'AIS': {
        'mmgroundingdino_url': "https://api-code.ais.uni-bonn.de/v1/vision/mmgroundingdino",
        'mmgroundingdino_key_type': "environment",
        'mmgroundingdino_key_value': "NIMBRO_VISION_API_KEY",

        'sam2_realtime_url': "https://api-code.ais.uni-bonn.de/v1/vision/sam2_realtime",
        'sam2_realtime_key_type': "environment",
        'sam2_realtime_key_value': "NIMBRO_VISION_API_KEY",

        'dam_url': "https://api-code.ais.uni-bonn.de/v1/vision/dam",
        'dam_key_type': "environment",
        'dam_key_value': "NIMBRO_VISION_API_KEY",

        'florence2_url': "https://api-code.ais.uni-bonn.de/v1/vision/florence2",
        'florence2_key_type': "environment",
        'florence2_key_value': "NIMBRO_VISION_API_KEY",

        'kosmos2_url': "https://api-code.ais.uni-bonn.de/v1/vision/kosmos2",
        'kosmos2_key_type': "environment",
        'kosmos2_key_value': "NIMBRO_VISION_API_KEY",
    }
}

### </Parameter Defaults>

class NimbroVision(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # initialize endpoints

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

        self.filter_api_endpoint = filter_api_endpoint.__get__(self)

        self.api_endpoints = api_endpoints
        self.endpoint_probes = {}
        self.model_locks = {}

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
            description="Probes the API endpoint to validate the API key and model name.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="probe_model_state",
            dtype=bool,
            default_value=probe_model_state,
            description="Probes the model state before inference and loads the requested model if required.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="api_endpoint",
            dtype=str,
            default_value=api_endpoint,
            description=f"Sets the API endpoint defining URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}.",
            read_only=False
        )

        # create interfaces

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)

        self.srv_mmgd = self.create_service(NimbroVisionGet, f"{self.node_namespace}/{self.node_name}/mmgroundingdino".replace("//", "/"), self.mmgroundingdino_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_sam2_realtime_update = self.create_service(NimbroVisionGet, f"{self.node_namespace}/{self.node_name}/sam2_realtime_update".replace("//", "/"), self.sam2_realtime_update_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_sam2_realtime_track = self.create_service(NimbroVisionGet, f"{self.node_namespace}/{self.node_name}/sam2_realtime_track".replace("//", "/"), self.sam2_realtime_track_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_dam = self.create_service(NimbroVisionGet, f"{self.node_namespace}/{self.node_name}/dam".replace("//", "/"), self.dam_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_kosmos2 = self.create_service(NimbroVisionGet, f"{self.node_namespace}/{self.node_name}/kosmos2".replace("//", "/"), self.kosmos2_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_florence2 = self.create_service(NimbroVisionGet, f"{self.node_namespace}/{self.node_name}/florence2".replace("//", "/"), self.florence2_callback, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())

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
            for model in self.model_locks:
                self.model_locks[model].acquire()
            value, message = self.filter_api_endpoint(name, value, line_length)
            if value is None:
                for model in self.model_locks:
                    self.model_locks[model].release()
            else:
                models = [key[:-4] for key in self.api_endpoints[value] if key[-4:] == "_url"]
                self._logger.debug(f"Model types featured by endpoint '{value}': {models}")
                model_locks = {}
                for model in models:
                    model_locks[model] = threading.Lock()
                self.model_locks = model_locks

        return value, message

    def validate_api_endpoint(self, api_endpoint):
        try:
            json_object = json.loads(api_endpoint)
        except Exception:
            success = False
            message = f"Value must be a valid JSON encoded dictionary containing a new endpoint or a name of an existing endpoint: {list(self.api_endpoints.keys())}"
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
                            message = None

        return success, message, json_object

    # API requests

    def probe_models_api(self, api_endpoint, model):
        api_endpoint_name = api_endpoint
        api_endpoint = self.api_endpoints[api_endpoint]

        success = True

        if api_endpoint[f"{model}_key_type"] == "environment":
            var_name = api_endpoint[f"{model}_key_value"]
            api_key = os.getenv(var_name)
            if api_key is None:
                success = False
                message = f"Failed to read API key from environment variable '{var_name}'."
        else:
            api_key = api_endpoint[f"{model}_key_value"]

        if success:
            url = f"{api_endpoint[model + '_url']}/model_flavors"
            self._logger.debug(f"Probing Models API of model '{model}' using URL '{url}' of endpoint '{api_endpoint_name}' using key '{api_key}'")
            try:
                response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
            except Exception as e:
                success = False
                message = f"Failed to retrieve available '{model}' models: {repr(e)}"
            else:
                if response.status_code != 200:
                    success = False
                    message = f"Received unexpected HTTP status code '{response.status_code}' from Models API: {response.text}"
                    message = remove_whitespace(string=message, reduce_to_single_space=True)
                else:
                    # self._logger.debug(f"{json.dumps(response.json(), indent=2)}")
                    available_models = response.json().get('flavors', [])
                    if len(available_models) == 0:
                        message = f"There are no '{model}' model flavors served under API endpoint '{api_endpoint_name}'."
                        self._logger.warn(message)
                    else:
                        message = f"'{model}' models flavors served under API endpoint '{api_endpoint_name}': {available_models}."
                        self._logger.debug(message)
                        if api_endpoint_name not in self.endpoint_probes:
                            self.endpoint_probes[api_endpoint_name] = {}
                        self.endpoint_probes[api_endpoint_name][model] = {'models': available_models, 'stamp': time.time()}

        if not success:
            self._logger.error(message)
            if api_endpoint_name in self.endpoint_probes:
                if model in self.endpoint_probes[api_endpoint_name]:
                    del self.endpoint_probes[api_endpoint_name][model]
                if len(self.endpoint_probes[api_endpoint_name]) == 0:
                    del self.endpoint_probes[api_endpoint_name]

        return success, message

    def validate_connection(self, model, flavor):
        probe = False
        if self.parameters.api_endpoint in self.endpoint_probes:
            if model in self.endpoint_probes[self.parameters.api_endpoint]:
                if flavor in self.endpoint_probes[self.parameters.api_endpoint][model]['models']:
                    success = True
                    message = f"Model '{model}' serves flavor '{flavor}' under API endpoint '{self.parameters.api_endpoint}'."
                    self._logger.debug(message)
                else:
                    success = False
                    message = f"Model '{model}' is not serving flavor '{flavor}' under API endpoint '{self.parameters.api_endpoint}': {self.endpoint_probes[self.parameters.api_endpoint][model]['models']}"
                    self._logger.error(message)
            else:
                probe = True
        else:
            probe = True

        if probe:
            success, message = self.probe_models_api(api_endpoint=self.parameters.api_endpoint, model=model)
            if success:
                if self.parameters.api_endpoint in self.endpoint_probes:
                    if model in self.endpoint_probes[self.parameters.api_endpoint]:
                        if flavor in self.endpoint_probes[self.parameters.api_endpoint][model]['models']:
                            success = True
                            message = f"Model '{model}' serves flavor '{flavor}' under API endpoint '{self.parameters.api_endpoint}'."
                            self._logger.debug(message)
                        else:
                            success = False
                            message = f"Model '{model}' is not serving flavor '{flavor}' under API endpoint '{self.parameters.api_endpoint}': {self.endpoint_probes[self.parameters.api_endpoint][model]['models']}"
                            self._logger.error(message)
                    else:
                        success = False
                else:
                    success = False

        return success, message

    def handle_request(self, model, request, response):
        self._logger.debug(f"handle_request('{model}'): start")
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

        # resolve model name
        if request.model_id > 0:
            model = f"{model}_{request.model_id}"

        # check request
        if response.success:
            response.success = False
            try:
                data = json.loads(request.data)
            except Exception as e:
                response.message = f"Failed to parse request field 'data' as JSON: {repr(e)}"
            else:
                response.success = True
                self._logger.debug("Parsed request data as JSON")

        # encode images if required
        if response.success:
            response.success = False
            if isinstance(data.get('image'), str):
                if is_base64(data['image']):
                    response.success = True
                    self._logger.debug("Image is Base64-encoded")
                elif is_url(data['image']):
                    response.success = True
                    self._logger.debug("Image is a valid URL")
                elif os.path.exists(data['image']):
                    if os.path.isfile(data['image']):
                        success, message, base64_image = read_as_b64(
                            file_path=data['image'],
                            name="image",
                            logger=self._logger
                        )
                        if success:
                            response.success = True
                            data['image'] = base64_image
                        else:
                            response.message = f"Failed to Base64-encode image file '{data['image']}': {message}"
                    else:
                        response.message = f"Failed to Base64-encode image file '{data['image']}' because it is a folder."
                else:
                    response.message = f"Image '{data['image']}' is not Base64-encoded, a valid local path or a web URL."
            elif isinstance(data.get('images'), list) and all(isinstance(image, str) for image in data['images']):
                for i, image_path in enumerate(data['images']):
                    if is_base64(image_path):
                        self._logger.debug(f"Image '{i + 1}' of '{len(data['images'])}' is Base64-encoded")
                    elif is_url(image_path):
                        response.success = True
                        self._logger.debug(f"Image '{i + 1}' of '{len(data['images'])}' is a valid URL")
                    elif os.path.exists(image_path):
                        if os.path.isfile(image_path):
                            success, message, base64_image = read_as_b64(
                                file_path=image_path,
                                name=f"image '{i + 1}' of '{len(data['images'])}'",
                                logger=self._logger
                            )
                            if success:
                                response.success = True
                                data['images'][i] = base64_image
                            else:
                                response.message = f"Failed to Base64-encode image file '{image_path}': {message}"
                                break
                        else:
                            response.message = f"Failed to Base64-encode image file '{image_path}' because it is a folder."
                            break
                    else:
                        response.message = f"Image '{image_path}' is neither Base64-encoded, a valid local path, or a web URL."
                        break
                else:
                    response.success = True
            else:
                response.success = True

        # check if model is defined in endpoint
        if model not in self.model_locks:
            response.success = False
            response.message = f"Model '{model}' is served under API endpoint '{self.parameters.api_endpoint}': {list(self.model_locks.keys())}"

        # lock model
        if response.success:
            if self.model_locks[model].locked():
                self._logger.info(f"Waiting for model '{model}' to be released before using it")
            self.model_locks[model].acquire()

        # validate connection
        if response.success and self.parameters.probe_api_connection and not sam_track:
            response.success, response.message = self.validate_connection(model=model, flavor=request.flavor)

        # retrieve API key
        if response.success:
            response.success = False
            url = self.api_endpoints[self.parameters.api_endpoint][f"{model}_url"]
            if self.api_endpoints[self.parameters.api_endpoint][f"{model}_key_type"] == "environment":
                api_key = os.getenv(self.api_endpoints[self.parameters.api_endpoint][f"{model}_key_value"])
                if api_key is None:
                    var = self.api_endpoints[self.parameters.api_endpoint][f"{model}_key_value"]
                    response.message = f"Failed to read API key from environment variable '{var}'."
                else:
                    response.success = True
            else:
                response.success = True
                api_key = self.api_endpoints[self.parameters.api_endpoint][f"{model}_key_value"]
            if api_key is not None:
                self._logger.debug(f"Retrieved '{model}' key '{api_key}'")

        # retrieve loaded state/flavor
        if response.success and not sam_track:
            if self.parameters.probe_model_state:

                response.success = False
                self._logger.debug(f"Requesting '{model}' status via URL '{url}/status'")
                try:
                    result = requests.get(f"{url}/status", headers={"Authorization": f"Bearer {api_key}"})
                except Exception as e:
                    response.message = f"Failed to get status request: {repr(e)}"
                else:
                    if result.status_code == 200:
                        load = False
                        result = result.json()
                        if model.find(result.get('model_family')) == -1:
                            response.message = f"URL '{url}' hosts wrong model type '{result['model_family']}'."
                        else:
                            response.success = True
                            if result.get('status') is None:
                                load = True
                                self._logger.debug(f"URL '{url}' currently hosts no model flavor instead of '{request.flavor}'")
                            elif result.get('status', {}).get('flavor') != request.flavor:
                                load = True
                                self._logger.debug(f"URL '{url}' currently hosts model flavor '{result.get('status', {}).get('flavor')}' instead of '{request.flavor}'")
                            else:
                                self._logger.debug(f"URL '{url}' currently hosts the requested model flavor '{request.flavor}'")
                    else:
                        response.message = f"Status request failed with status code '{result.status_code}': {result.text}"

                # load requested model flavor if required
                if response.success and load:
                    response.success = False
                    self._logger.info(f"Loading '{model}' flavor '{request.flavor}'")
                    try:
                        result = requests.post(f"{url}/load", json={'flavor': request.flavor}, headers={"Authorization": f"Bearer {api_key}"})
                    except Exception as e:
                        response.message = f"Failed to POST load request: {repr(e)}"
                    else:
                        if result.status_code == 200:
                            self._logger.info(f"Loaded '{model}' flavor '{request.flavor}': {result.text}")
                            response.success = True
                        else:
                            response.message = f"Load request failed with status code '{result.status_code}': {result.text}"

        # post inference request
        if response.success:
            response.success = False
            if model == "sam2_realtime" and not sam_track:
                suffix = "update"
            else:
                suffix = "infer"
            self._logger.debug(f"Starting '{model}' inference via URL '{url}/{suffix}'")
            try:
                result = requests.post(f"{url}/{suffix}", json=data, headers={"Authorization": f"Bearer {api_key}"})
            except Exception as e:
                response.message = f"Failed to POST inference request: {repr(e)}"
            else:
                if result.status_code == 200:
                    response.result = result.content.decode('utf-8')
                    response.success = True
                    duration = time.perf_counter() - stamp_start
                    self._logger.debug(f"Response: {response.result[:80]}{'...' if len(result.text) >= 80 else ''}")
                    if sam_track:
                        response.message = f"Retrieved tracking response from model '{model}' after '{duration:.3f}s'."
                    else:
                        response.message = f"Retrieved {'inference' if suffix == 'infer' else 'update'} response from model '{model}' with flavor '{request.flavor}' after '{duration:.3f}s'."
                else:
                    if sam_track:
                        response.message = f"Tracking failed with status code '{result.status_code}': {result.text}"
                    else:
                        response.message = f"{'Inference' if suffix == 'infer' else 'Update'} failed with status code '{result.status_code}': {result.text}"

        # log
        if response.success:
            self._logger.info(response.message)
        else:
            self._logger.error(f"Error occurred while using '{model}': {response.message}")

        # release model
        if model in self.model_locks and self.model_locks[model].locked():
            self.model_locks[model].release()

        self._logger.debug(f"handle_request('{model}'): end after '{time.perf_counter() - stamp_start:.3f}s'")
        return response

    # service callbacks

    def mmgroundingdino_callback(self, request, response):
        try:
            response = self.handle_request('mmgroundingdino', request, response)
        except Exception as e:
            self._logger.error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def sam2_realtime_update_callback(self, request, response):
        try:
            response = self.handle_request('sam2_realtime_update', request, response)
        except Exception as e:
            self._logger.error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def sam2_realtime_track_callback(self, request, response):
        try:
            response = self.handle_request('sam2_realtime_track', request, response)
        except Exception as e:
            self._logger.error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def dam_callback(self, request, response):
        try:
            response = self.handle_request('dam', request, response)
        except Exception as e:
            self._logger.error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def kosmos2_callback(self, request, response):
        try:
            response = self.handle_request('kosmos2', request, response)
        except Exception as e:
            self._logger.error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

    def florence2_callback(self, request, response):
        try:
            response = self.handle_request('florence2', request, response)
        except Exception as e:
            self._logger.error(f"{type(e).__name__}: {repr(e)}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"Unexpected error: {repr(e)}"
        return response

def main(args=None):
    start_and_spin_node(NimbroVision, args=args)

if __name__ == '__main__':
    main()
