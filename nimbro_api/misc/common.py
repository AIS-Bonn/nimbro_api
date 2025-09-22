#!/usr/bin/env python3

import os
import copy
import json
import time

import requests

from nimbro_utils.lazy import log_lines, remove_whitespace

def validate_default_endpopints(self, api_endpoints):
    assert isinstance(api_endpoints, dict), f"Expected 'api_endpoints' to be of type 'dict' instead of '{type(api_endpoints).__name__}'."
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

def filter_api_endpoint(self, name, value, line_length):
    message = None

    if value not in self.api_endpoints:
        success, message, json_object = self.validate_api_endpoint(value)
        if success:
            enpoint_name = json_object['name']
            json_object_without_name = copy.deepcopy(json_object)
            del json_object_without_name['name']
            if enpoint_name in self.api_endpoints:
                if json_object_without_name == self.api_endpoints[enpoint_name]:
                    message = f"Kept API endpoint '{value}'."
                    self._logger.debug(message)
                else:
                    if self._logger.get_severity() > 10:
                        self._logger.info(f"Updated API endpoint '{enpoint_name}'.")
                    log_lines(
                        text=f"Updated API endpoint '{enpoint_name}':\n{json.dumps(json_object_without_name, indent=2)}",
                        line_length=line_length,
                        line_highlight="| ",
                        block_format=False,
                        allow_empty_lines=True,
                        logger=self._logger,
                        severity=10
                    )
                    if enpoint_name in self.endpoint_probes:
                        del self.endpoint_probes[enpoint_name]
                        self._logger.debug(f"Deleted existing model probe of updated API endpoint '{enpoint_name}'")
            else:
                if self._logger.get_severity() > 10:
                    self._logger.info(f"Created API endpoint '{enpoint_name}'.")
                log_lines(
                    text=f"Created API endpoint '{enpoint_name}':\n{json.dumps(json_object_without_name, indent=2)}",
                    line_length=line_length,
                    line_highlight="| ",
                    block_format=False,
                    allow_empty_lines=True,
                    logger=self._logger,
                    severity=10
                )

            self.api_endpoints[enpoint_name] = json_object_without_name
            value = enpoint_name
        else:
            value = None

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
            message = None

    return success, message, json_object

def retrieve_api_key(self):
    if self.api_endpoints[self.parameters.api_endpoint]['key_type'] == "environment":
        api_key = os.getenv(self.api_endpoints[self.parameters.api_endpoint]['key_value'])
        if api_key is None:
            message = f"Failed to read API key from environment variable '{self.api_endpoints[self.parameters.api_endpoint]['key_value']}'."
            return False, message, None
    else:
        api_key = self.api_endpoints[self.parameters.api_endpoint]['key_value']
    message = f"Retrieved API key '{api_key}'."
    self._logger.debug(message)
    return True, message, api_key

def probe_models_api(self, api_endpoint):
    api_endpoint_name = api_endpoint
    api_endpoint = self.api_endpoints[api_endpoint]

    success = True

    if 'models_url' not in api_endpoint:
        message = f"Cannot probe Models API because endpoint '{api_endpoint_name}' does not contain a Models URL."
        self._logger.warn(message)
    else:
        if api_endpoint['key_type'] == "environment":
            api_key = os.getenv(api_endpoint['key_value'])
            if api_key is None:
                success = False
                message = f"Failed to read API key from environment variable '{api_endpoint['key_value']}'."
        else:
            api_key = api_endpoint['key_value']

        if success:
            self._logger.debug(f"Probing Models API '{api_endpoint['models_url']}' of endpoint '{api_endpoint_name}' using key '{api_key}'")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            try:
                response = requests.get(api_endpoint['models_url'], headers=headers)
            except Exception as e:
                success = False
                message = f"Failed to retrieve available models: {repr(e)}"
            else:
                if response.status_code != 200:
                    success = False
                    message = f"Received unexpected HTTP status code '{response.status_code}' from Models API: {response.text}"
                    message = remove_whitespace(string=message, reduce_to_single_space=True)
                else:
                    # self._logger.debug(f"{json.dumps(response.json(), indent=2)}")
                    available_models = [m['id'] for m in response.json()['data']]
                    if len(available_models) == 0:
                        message = f"There are no models served under API endpoint '{api_endpoint_name}'."
                        self._logger.warn(message)
                    else:
                        message = f"Models served under API endpoint '{api_endpoint_name}': {available_models}."
                        self._logger.debug(message)
                        self.endpoint_probes[api_endpoint_name] = {'models': available_models, 'stamp': time.time()}

    if not success:
        self._logger.error(message)
        if api_endpoint_name in self.endpoint_probes:
            del self.endpoint_probes[api_endpoint_name]

    return success, message

def validate_connection(self, model):
    if self.parameters.api_endpoint in self.endpoint_probes:
        if model.split(":")[0] in self.endpoint_probes[self.parameters.api_endpoint]['models']:
            success = True
            message = f"Model '{model}' is served under API endpoint '{self.parameters.api_endpoint}'."
            self._logger.debug(message)
        else:
            success = False
            message = f"Model '{model}' is not served under API endpoint '{self.parameters.api_endpoint}': {self.endpoint_probes[self.parameters.api_endpoint]['models']}"
            self._logger.error(message)
    else:
        success, message = self.probe_models_api(api_endpoint=self.parameters.api_endpoint)
        if success:
            if self.parameters.api_endpoint in self.endpoint_probes:
                if model.split(":")[0] in self.endpoint_probes[self.parameters.api_endpoint]['models']:
                    success = True
                    message = f"Model '{model}' is served under API endpoint '{self.parameters.api_endpoint}'."
                    self._logger.debug(message)
                else:
                    success = False
                    message = f"Model '{model}' is not served under API endpoint '{self.parameters.api_endpoint}': {self.endpoint_probes[self.parameters.api_endpoint]['models']}"
                    self._logger.error(message)
            else:
                success = False

    return success, message

class CustomException(Exception):
    pass
