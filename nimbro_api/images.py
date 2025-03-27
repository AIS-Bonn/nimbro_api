#!/usr/bin/env python3

import os
import json
import time
import base64
import datetime

import cv2
import requests
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_prefix

from nimbro_api_interfaces.srv import GetImage
from nimbro_api.utils.node import start_and_spin_node
from nimbro_api.utils.parameter_handler import ParameterHandler

### <Parameter Defaults>

node_name = "images"
logger_level = 10
api_flavor = "openai"

cache_read = True
cache_write = True
cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache", "images")
cache_file = "cache_images.json"

## non-params

api_settings = { # TODO switch to endpoints concept as in completions/embeddings
    'openai': {
        'key_variable': "OPENAI_API_KEY",
        'models_url': "https://api.openai.com/v1/models",
        'images_url': "https://api.openai.com/v1/images/generations"
    }
}

### </Parameter Defaults>

class Images(Node):

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
        descriptor.description = "Attempt to retrieve images from cached results."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_read, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_write"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Cache retrieved images locally."
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

        self.cv_bridge = CvBridge()

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)

        self.cbg_image = ReentrantCallbackGroup()
        self.srv_image = self.create_service(GetImage, f"{self.node_namespace}/{self.node_name}/get_image".replace("//", "/"), self.get_image_callack, qos_profile=qos_profile, callback_group=self.cbg_image)

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

    # Images Pipeline

    def image_post(self, prompt, model, quality, style, size, api_url, api_key):
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }

        if model == "dall-e-3":
            data = {
                'prompt': prompt,
                'model': model,
                'quality': quality,
                'style': style,
                'size': size,
                'n': 1,
                'response_format': "b64_json"
            }
        elif model == "dall-e-2":
            data = {
                'prompt': prompt,
                'model': model,
                'size': size,
                'n': 1,
                'response_format': "b64_json"
            }
        else:
            assert False, f"Unsupported model name '{model}'"

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
            response = response.json()
            success = True
            message = "Successfully retrieved image."
            b64_image = response['data'][0]['b64_json']
        else:
            success = False
            message = f"HTTP-Error: {response.text}"
            b64_image = None

        return success, message, b64_image

    def get_image(self, prompt, model, quality, style, size):
        # parse arguments

        # TODO normalize prompt

        if prompt == "":
            message = "Cannot generate image for empty prompt."
            self.get_logger().error(message)
            return False, message, None, None

        supported_models = ["dall-e-3", "dall-e-2"]
        if model == "":
            model = supported_models[0]
            self.get_logger().debug(f"Using default model '{model}'")

        if model not in supported_models:
            message = f"Model '{model}' is not supported. Supported models are: {supported_models}"
            self.get_logger().error(message)
            return False, message, None, None

        if model not in self.available_models:
            message = f"Model '{model}' is not available. Available models are: {self.available_models}"
            self.get_logger().error(message)
            return False, message, None, None

        if model == "dall-e-3":

            supported_sizes = ["1792x1024", "1024x1792", "1024x1024"]
            if size == "":
                size = supported_sizes[0]
                self.get_logger().debug(f"Using default size '{size}'")

            if size not in supported_sizes:
                message = f"Size '{size}' is not supported. Supported sizes are: {supported_sizes}"
                self.get_logger().error(message)
                return False, message, None, None

            supported_qualities = ["hd", "standard"]
            if quality == "":
                quality = supported_qualities[0]
                self.get_logger().debug(f"Using default quality '{quality}'")

            if quality not in supported_qualities:
                message = f"Quality '{quality}' is not supported. Supported qualities are: {supported_qualities}"
                self.get_logger().error(message)
                return False, message, None, None

            supported_styles = ["vivid", "natural"]
            if style == "":
                style = supported_styles[0]
                self.get_logger().debug(f"Using default style '{style}'")

            if style not in supported_styles:
                message = f"Style '{style}' is not supported. Supported styles are: {supported_styles}"
                self.get_logger().error(message)
                return False, message, None, None

        elif model == "dall-e-2":

            supported_sizes = ["1024x1024", "512x512", "256x256"]
            if size == "":
                size = supported_sizes[0]
                self.get_logger().debug(f"Using default size '{size}'")

            if size not in supported_sizes:
                message = f"Size '{size}' is not supported. Supported sizes are: {supported_sizes}"
                self.get_logger().error(message)
                return False, message, None, None

        else:
            assert False, f"Unsupported model name '{model}'"

        # read cache

        image_cv = None

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
                    if model == "dall-e-3":
                        image_path = cache.get(model, {}).get(size, {}).get(quality, {}).get(style, {}).get(prompt)
                    elif model == "dall-e-2":
                        image_path = cache.get(model, {}).get(size, {}).get(prompt)
                    else:
                        assert False, f"Unsupported model name '{model}'"
                    if image_path is None:
                        self.get_logger().debug("Image not found in cache")
                    else:
                        self.get_logger().debug(f"Reading image '{image_path}' found in cache")
                        try:
                            image_cv = cv2.imread(image_path)
                        except Exception as e:
                            self.get_logger().error(f"Failed to read cached image '{image_path}': {e}")
                            image_cv = None

        # generate image if necessary

        if image_cv is None:
            # post to API

            self.get_logger().info(f"Retrieving image from API (prompt='{prompt}', model='{model}', quality='{quality}', style='{style}', size='{size}')")
            success, message, image_b64 = self.image_post(
                prompt=prompt,
                model=model,
                quality=quality,
                style=style,
                size=size,
                api_url=api_settings[api_flavor]['images_url'],
                api_key=self.api_key
            )
            if not success:
                self.get_logger().error(message)
                return False, message, None, None

            # convert base64 image to cv2

            self.get_logger().debug("Converting base64 image to cv2")
            try:
                image_np = np.frombuffer(base64.b64decode(image_b64), np.uint8)
                image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            except Exception as e:
                message = f"Failed to convert base64 image to cv2: {e}"
                self.get_logger().error(message)
                return False, message, None, None

            # write image to file

            stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            image_path = os.path.join(self.cache_folder, f"{stamp}.png")
            self.get_logger().debug(f"Writing image to file '{image_path}'")

            try:
                cv2.imwrite(image_path, cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))
                if not os.path.isfile(image_path):
                    raise Exception("Unknown error not reported by OpenCV")
            except Exception as e:
                message = f"Failed to write image to file '{image_path}': {e}"
                self.get_logger().error(message)
                return False, message, None

            # write path to cache

            if cache_write:
                cache_path = os.path.join(self.cache_folder, self.cache_file)
                self.get_logger().debug(f"Writing image path to cache file '{cache_path}'")

                # add path to cache

                if model not in cache:
                    cache[model] = {}

                if size not in cache[model]:
                    cache[model][size] = {}

                if model == "dall-e-3":
                    if quality not in cache[model][size]:
                        cache[model][size][quality] = {}
                    if style not in cache[model][size][quality]:
                        cache[model][size][quality][style] = {}
                    cache[model][size][quality][style][prompt] = image_path
                elif model == "dall-e-2":
                    cache[model][size][prompt] = image_path
                else:
                    assert False, f"Unsupported model name '{model}'"

                # create cache folder

                if not os.path.exists(self.cache_folder):
                    self.get_logger().debug(f"Creating cache folder '{self.cache_folder}'")
                    try:
                        os.makedirs(self.cache_folder)
                    except Exception as e:
                        self.get_logger().error(f"Failed to create cache folder '{self.cache_folder}': {e}")

                # write cache

                if os.path.exists(self.cache_folder):
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump(cache, f, indent=4)
                    except Exception as e:
                        self.get_logger().error(f"Failed to save image path to cache file '{cache_path}': {e}")

        # convert cv2 image to ROS

        self.get_logger().debug("Converting cv2 image to ROS")
        try:
            image_msg = self.cv_bridge.cv2_to_imgmsg(image_cv, encoding="rgb8")
        except Exception as e:
            message = f"Failed to convert image from cv2: {e}"
            self.get_logger().error(message)
            return False, message, None, None

        # forward results

        self.get_logger().info(f"Successfully retrieved image '{image_path}' (prompt='{prompt}', model='{model}', quality='{quality}', style='{style}', size='{size}')")

        return True, "Successfully retrieved image.", image_msg, image_path

    # Callbacks

    def get_image_callack(self, request, response):
        self.get_logger().debug("get_image_callack(): start")

        response.success, response.message, image_msg, image_path = self.get_image(
            prompt=request.prompt,
            model=request.model,
            quality=request.quality,
            style=request.style,
            size=request.size
        )
        if response.success:
            response.image = image_msg
            response.path = image_path

        self.get_logger().debug("get_image_callack(): end")
        return response

def main(args=None):
    start_and_spin_node(Images, args=args)

if __name__ == '__main__':
    main()
