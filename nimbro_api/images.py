#!/usr/bin/env python3

import os
import json
import time
import datetime

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from ament_index_python.packages import get_package_prefix

from nimbro_api_interfaces.srv import ImagesGet
from nimbro_api.misc.common import validate_default_endpopints, filter_api_endpoint, validate_api_endpoint, retrieve_api_key, probe_models_api, validate_connection

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, decode_b64, read_as_b64

### <Parameter Defaults>

node_name = "images"
severity = 10

probe_api_connection = True
api_endpoint = "OpenAI"

cache_read = True
cache_write = True
cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "images")
cache_file = "cache_images.json"

## non-params

line_length = 150

api_endpoints = {
    'OpenAI': {
        'api_flavor': "openai",
        'models_url': "https://api.openai.com/v1/models",
        'images_url': "https://api.openai.com/v1/images/generations",
        'key_type': "environment",
        'key_value': "OPENAI_API_KEY"
    }
}

### </Parameter Defaults>

class Images(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # initialize endpoints

        self.endpoint_keys_required = {'name', 'api_flavor', 'images_url', 'key_type', 'key_value'}
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
            description=f"Sets the API endpoint defining API flavor, Models & Images URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_read",
            dtype=bool,
            default_value=cache_read,
            description="Attempt to retrieve images from cached results.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_write",
            dtype=bool,
            default_value=cache_write,
            description="Cache retrieved images locally.",
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

        self.cbg_image = ReentrantCallbackGroup()
        self.srv_image = self.create_service(ImagesGet, f"{self.node_namespace}/{self.node_name}/get_image".replace("//", "/"), self.get_image_callack, qos_profile=qos_profile, callback_group=self.cbg_image)

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
                value = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache")

        return value, message

    # Images Pipeline

    def image_post(self, prompt, model, quality, style, size, api_url, api_key):
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }

        if model == "gpt-image-1":
            data = {
                'prompt': prompt,
                'model': model,
                'quality': quality,
                'size': size,
                'background': "auto",
                'n': 1,
                'moderation': "low",
                'output_format': "png"
            }
        elif model == "dall-e-3":
            data = {
                'prompt': prompt,
                'model': model,
                'quality': quality,
                'style': style,
                'size': size,
                'n': 1,
                'output_format': "png",
                'response_format': "b64_json"
            }
        elif model == "dall-e-2":
            data = {
                'prompt': prompt,
                'model': model,
                'size': size,
                'n': 1,
                'output_format': "png",
                'response_format': "b64_json"
            }
        else:
            assert False, f"Unsupported model name '{model}'."

        self._logger.debug(f"Posting request: {data}")
        tic = time.perf_counter()
        response = requests.post(api_url, headers=headers, json=data, stream=False)
        toc = time.perf_counter()
        self._logger.debug(f"Received response after '{toc - tic:.3f}s'")

        if response.status_code == 200:
            response = response.json()
            success = True
            message = "Retrieved image."
            b64_image = response['data'][0]['b64_json']
        else:
            success = False
            message = f"HTTP-Error: {response.text}"
            b64_image = None

        return success, message, b64_image

    def get_image(self, prompt, model, quality, style, size):
        # parse arguments

        if prompt == "":
            message = "Cannot generate image for empty prompt."
            self._logger.error(message)
            return False, message, None

        supported_models = ["dall-e-3", "dall-e-2", "gpt-image-1"]
        if model == "":
            model = supported_models[0]
            self._logger.debug(f"Using default model '{model}'")

        if model not in supported_models:
            message = f"Model '{model}' is not supported. Supported models are: {supported_models}"
            self._logger.error(message)
            return False, message, None

        if model == "gpt-image-1":

            supported_sizes = ["1536x1024", "1024x1536", "1024x1024"]
            if size == "":
                size = supported_sizes[0]
                self._logger.debug(f"Using default size '{size}'")

            if size not in supported_sizes:
                message = f"Size '{size}' is not supported. Supported sizes are: {supported_sizes}"
                self._logger.error(message)
                return False, message, None

            supported_qualities = ["high", "medium", "low"]
            if quality == "":
                quality = supported_qualities[0]
                self._logger.debug(f"Using default quality '{quality}'")

            if quality not in supported_qualities:
                message = f"Quality '{quality}' is not supported. Supported qualities are: {supported_qualities}"
                self._logger.error(message)
                return False, message, None

            supported_styles = [""]
            if style == "":
                style = supported_styles[0]
                self._logger.debug(f"Using default style '{style}'")

            if style not in supported_styles:
                message = f"Style '{style}' is not supported. Supported styles are: {supported_styles}"
                self._logger.error(message)
                return False, message, None

        elif model == "dall-e-3":

            supported_sizes = ["1792x1024", "1024x1792", "1024x1024"]
            if size == "":
                size = supported_sizes[0]
                self._logger.debug(f"Using default size '{size}'")

            if size not in supported_sizes:
                message = f"Size '{size}' is not supported. Supported sizes are: {supported_sizes}"
                self._logger.error(message)
                return False, message, None

            supported_qualities = ["hd", "standard"]
            if quality == "":
                quality = supported_qualities[0]
                self._logger.debug(f"Using default quality '{quality}'")

            if quality not in supported_qualities:
                message = f"Quality '{quality}' is not supported. Supported qualities are: {supported_qualities}"
                self._logger.error(message)
                return False, message, None

            supported_styles = ["vivid", "natural"]
            if style == "":
                style = supported_styles[0]
                self._logger.debug(f"Using default style '{style}'")

            if style not in supported_styles:
                message = f"Style '{style}' is not supported. Supported styles are: {supported_styles}"
                self._logger.error(message)
                return False, message, None

        elif model == "dall-e-2":

            supported_sizes = ["1024x1024", "512x512", "256x256"]
            if size == "":
                size = supported_sizes[0]
                self._logger.debug(f"Using default size '{size}'")

            if size not in supported_sizes:
                message = f"Size '{size}' is not supported. Supported sizes are: {supported_sizes}"
                self._logger.error(message)
                return False, message, None

            supported_qualities = ["standard"]
            if quality == "":
                quality = supported_qualities[0]
                self._logger.debug(f"Using default quality '{quality}'")

            if quality not in supported_qualities:
                message = f"Quality '{quality}' is not supported. Supported qualities are: {supported_qualities}"
                self._logger.error(message)
                return False, message, None

            supported_styles = [""]
            if style == "":
                style = supported_styles[0]
                self._logger.debug(f"Using default style '{style}'")

            if style not in supported_styles:
                message = f"Style '{style}' is not supported. Supported styles are: {supported_styles}"
                self._logger.error(message)
                return False, message, None

        else:
            assert False, f"Unsupported model name '{model}'."

        # read cache

        image_b64 = None

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
                    if model == "gpt-image-1":
                        image_path = cache.get(model, {}).get(size, {}).get(quality, {}).get(prompt)
                    elif model == "dall-e-3":
                        image_path = cache.get(model, {}).get(size, {}).get(quality, {}).get(style, {}).get(prompt)
                    elif model == "dall-e-2":
                        image_path = cache.get(model, {}).get(size, {}).get(prompt)
                    else:
                        assert False, f"Unsupported model name '{model}'."
                    if image_path is None:
                        self._logger.debug("Image not found in cache")
                    else:
                        self._logger.debug(f"Found image '{image_path}' in cache")
                        _, _, image_b64 = read_as_b64(file_path=image_path, name="image", logger=self._logger)

        # generate image if necessary

        if image_b64 is None:
            # validate connection
            if self.parameters.probe_api_connection:
                success, message = self.validate_connection(model=model)
                if not success:
                    return False, message, None

            # retrieve API key
            success, message, api_key = self.retrieve_api_key()
            if not success:
                self._logger.error(message)
                return False, message, None

            # use API
            self._logger.info(f"Retrieving image from API (prompt='{prompt}', model='{model}', quality='{quality}', style='{style}', size='{size}')")
            success, message, image_b64 = self.image_post(
                prompt=prompt,
                model=model,
                quality=quality,
                style=style,
                size=size,
                api_url=self.api_endpoints[self.parameters.api_endpoint]['images_url'],
                api_key=api_key
            )
            if not success:
                self._logger.error(message)
                return False, message, None

            # decode Base64 image

            success, message, image_bytes = decode_b64(string=image_b64, name="image", logger=self._logger)
            if not success:
                return False, message, None

            # write image to file

            stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            image_path = os.path.join(self.parameters.cache_folder, f"{stamp}.png")
            self._logger.debug(f"Writing image to file '{image_path}'")
            try:
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                message = f"Failed to write image to file '{image_path}': {repr(e)}"
                self._logger.error(message)
                return False, message, None

            # write path to cache

            if cache_write:
                cache_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)
                self._logger.debug(f"Writing image path to cache file '{cache_path}'")

                # add path to cache

                if model not in cache:
                    cache[model] = {}

                if size not in cache[model]:
                    cache[model][size] = {}

                if model == "gpt-image-1":
                    if quality not in cache[model][size]:
                        cache[model][size][quality] = {}
                    cache[model][size][quality][prompt] = image_path
                if model == "dall-e-3":
                    if quality not in cache[model][size]:
                        cache[model][size][quality] = {}
                    if style not in cache[model][size][quality]:
                        cache[model][size][quality][style] = {}
                    cache[model][size][quality][style][prompt] = image_path
                elif model == "dall-e-2":
                    cache[model][size][prompt] = image_path
                else:
                    assert False, f"Unsupported model name '{model}'."

                # create cache folder

                if not os.path.exists(self.parameters.cache_folder):
                    self._logger.debug(f"Creating cache folder '{self.parameters.cache_folder}'")
                    try:
                        os.makedirs(self.parameters.cache_folder)
                    except Exception as e:
                        self._logger.error(f"Failed to create cache folder '{self.parameters.cache_folder}': {repr(e)}")

                # write cache

                if os.path.exists(self.parameters.cache_folder):
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump(cache, f, indent=4)
                    except Exception as e:
                        self._logger.error(f"Failed to save image path to cache file '{cache_path}': {repr(e)}")

        # forward results

        self._logger.info(f"Retrieved image '{image_path}' (prompt='{prompt}', model='{model}', quality='{quality}', style='{style}', size='{size}')")

        return True, "Retrieved image.", image_path

    # Callbacks

    def get_image_callack(self, request, response):
        self._logger.debug("get_image_callack(): start")

        response.success, response.message, image_path = self.get_image(
            prompt=request.prompt,
            model=request.model,
            quality=request.quality,
            style=request.style,
            size=request.size
        )
        if response.success:
            response.path = image_path

        self._logger.debug("get_image_callack(): end")
        return response

def main(args=None):
    start_and_spin_node(Images, args=args)

if __name__ == '__main__':
    main()
