#!/usr/bin/env python3

import os
import json
import time
import datetime
import threading

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange, FloatingPointRange

from nimbro_api_interfaces.msg import ApiUsage
from nimbro_api_interfaces.srv import GetUsage
from nimbro_api.utils.node import start_and_spin_node
from nimbro_api.utils.misc import read_json, write_json
from nimbro_api.utils.parameter_handler import ParameterHandler

### <Parameter Defaults>

node_name = "usage_monitor"
log_level = 10

cache_read_once = True
cache_write_lazy = True
cache_write_interval = 30.0
cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache")
cache_file = "cache_usage.json"

### </Parameter Defaults>

class UsageMonitor(Node):

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
        self.declare_parameter(descriptor.name, log_level, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_read_once"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Read usage cache file once when required and keep it in memory instead of loading it every time."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_read_once, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_write_lazy"
        descriptor.type = ParameterType.PARAMETER_BOOL
        descriptor.description = "Write usage cache file in fixed intervals instead of writing it with every update."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_write_lazy, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "cache_write_interval"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Minimum time in seconds in which the usage cache file is written if cache_write_lazy is active."
        descriptor.read_only = True
        float_range = FloatingPointRange()
        float_range.from_value = 10.0
        float_range.to_value = 3600.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, cache_write_interval, descriptor)

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

        self.define_prices()

        self.file_lock = threading.Lock()
        self.cache_write_required = False
        self.cache = None

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_ALL, depth=100)
        self.sub_usage = self.create_subscription(
            msg_type=ApiUsage,
            topic=f"{self.node_namespace}/api_usage".replace("//", "/"),
            callback=self.monitor_usage,
            qos_profile=qos_profile,
            callback_group=ReentrantCallbackGroup()
        )
        self.srv_get_usage = self.create_service(
            srv_type=GetUsage,
            srv_name=f"{self.node_namespace}/{self.node_name}/get_usage".replace("//", "/"),
            callback=self.get_usage,
            qos_profile=qos_profile,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.timer_write_lazy = self.create_timer(self.cache_write_interval, self.write_cache_lazy, callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info("Node started")

    def __del__(self):
        self.write_cache_lazy()
        self.get_logger().info("Node shutdown")

    def parameter_changed(self, parameter):
        success = True
        reason = ""

        if parameter.name == "logger_level":
            rclpy.logging.set_logger_level(f"{self.node_namespace}/{self.node_name}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(parameter.value))

        elif parameter.name == "cache_read_once":
            self.cache_read_once = parameter.value

        elif parameter.name == "cache_write_lazy":
            self.cache_write_lazy = parameter.value

        elif parameter.name == "cache_write_interval":
            self.cache_write_interval = parameter.value

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

    def define_prices(self):
        # TODO structure this by API type and flavor, put it in an external file, retrieve OpenRouter costs from Models API instead
        self.pricing = { # dollars per million tokens
            # Embeddings

            'text-embedding-3-small': {
                'input_tokens_uncached': 0.02,
            },
            'text-embedding-3-large': {
                'input_tokens_uncached': 0.13,
            },
            'text-embedding-ada-002': {
                'input_tokens_uncached': 0.1,
            },
            'mistral-embed': {
                'input_tokens_uncached': 0.1,
            },

            # Completions

            # OpenAI

            'gpt-4-1106-preview': {
                'input_tokens_uncached': 10.0,
                'output_tokens': 30.0
            },
            'gpt-4o-2024-11-20': {
                'input_tokens_uncached': 2.5,
                'input_tokens_cached': 1.25,
                'output_tokens': 1.25
            },
            'o1-mini': {
                'input_tokens_uncached': 3.0,
                'input_tokens_cached': 1.5,
                'output_tokens': 12.0
            },
            'o1-mini-2024-09-12': {
                'input_tokens_uncached': 3.0,
                'input_tokens_cached': 1.5,
                'output_tokens': 12.0
            },
            'o1-preview': {
                'input_tokens_uncached': 15.0,
                'input_tokens_cached': 7.50,
                'output_tokens': 60.0
            },
            'gpt-3.5-turbo-16k': {
                'input_tokens_uncached': 3.0,
                'output_tokens': 4.0
            },
            'o1-preview-2024-09-12': {
                'input_tokens_uncached': 15.0,
                'input_tokens_cached': 7.50,
                'output_tokens': 60.0
            },
            'gpt-4-0125-preview': {
                'input_tokens_uncached': 10.0,
                'output_tokens': 30.0
            },
            'gpt-4': {
                'input_tokens_uncached': 30.0,
                'output_tokens': 60.0
            },
            'gpt-4o-mini': {
                'input_tokens_uncached': 0.15,
                'input_tokens_cached': 0.075,
                'output_tokens': 0.6
            },
            'gpt-4o-mini-2024-07-18': {
                'input_tokens_uncached': 0.15,
                'input_tokens_cached': 0.075,
                'output_tokens': 0.6
            },
            'gpt-3.5-turbo': {
                'input_tokens_uncached': 0.5,
                'input_tokens_cached': 1.5,
            },
            'gpt-3.5-turbo-0125': {
                'input_tokens_uncached': 0.5,
                'output_tokens': 1.5
            },
            'gpt-4-turbo': {
                'input_tokens_uncached': 10.0,
                'output_tokens': 30.0
            },
            'gpt-4o': {
                'input_tokens_uncached': 2.5,
                'input_tokens_cached': 1.25,
                'output_tokens': 10.0
            },
            'gpt-4o-2024-08-06': {
                'input_tokens_uncached': 2.5,
                'input_tokens_cached': 1.25,
                'output_tokens': 10.0
            },
            'gpt-4-turbo-2024-04-09': {
                'input_tokens_uncached': 10.0,
                'output_tokens': 30.0
            },
            'gpt-3.5-turbo-1106': {
                'input_tokens_uncached': 1.0,
                'output_tokens': 2.0
            },
            'gpt-3.5-turbo-instruct': {
                'input_tokens_uncached': 1.5,
                'output_tokens': 2.0
            },
            'gpt-4o-audio-preview': {
                'input_tokens_uncached': 2.5,
                'output_tokens': 10.0
            },
            'gpt-4o-audio-preview-2024-10-01': {
                'input_tokens_uncached': 2.5,
                'output_tokens': 10.0
            },
            'gpt-3.5-turbo-instruct-0914': {
                'input_tokens_uncached': 1.5,
                'output_tokens': 2.0
            },
            'chatgpt-4o-latest': {
                'input_tokens_uncached': 5.0,
                'input_tokens_cached': 0.0,
                'output_tokens': 15.0
            },
            'gpt-4o-2024-05-13': {
                'input_tokens_uncached': 5.0,
                'output_tokens': 15.0
            },

            # Mistral AI

            'ministral-3b-2410': {
                'input_tokens_uncached': 0.04,
                'output_tokens': 0.04,
            },
            'ministral-3b-latest': {
                'input_tokens_uncached': 0.04,
                'output_tokens': 0.04,
            },
            'ministral-8b-2410': {
                'input_tokens_uncached': 0.1,
                'output_tokens': 0.1,
            },
            'ministral-8b-latest': {
                'input_tokens_uncached': 0.1,
                'output_tokens': 0.1,
            },
            'open-mistral-7b': {
                'input_tokens_uncached': 0.25,
                'output_tokens': 0.25,
            },
            'open-mistral-nemo': {
                'input_tokens_uncached': 0.15,
                'output_tokens': 0.15,
            },
            'open-mistral-nemo-2407': {
                'input_tokens_uncached': 0.15,
                'output_tokens': 0.15,
            },
            'open-mixtral-8x7b': {
                'input_tokens_uncached': 0.7,
                'output_tokens': 0.7,
            },
            'mistral-small': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.2,
            },
            'mistral-small-2312': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.2,
            },
            'open-mixtral-8x22b': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 6.0,
            },
            'open-mixtral-8x22b-2404': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 6.0,
            },
            'mistral-small-2402': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.2,
            },
            'mistral-small-2409': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.2,
            },
            'mistral-small-latest': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.2,
            },
            'mistral-large-2402': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 6.0,
            },
            'mistral-large-2407': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 6.0,
            },
            'mistral-large-2411': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 6.0,
            },
            'mistral-large-latest': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 6.0,
            },
            'pixtral-large-2411': {
                'input_tokens_uncached': 2.0,
                'input_tokens_cached': 2.0,
                'output_tokens': 6.0,
            },
            'pixtral-large-latest': {
                'input_tokens_uncached': 2.0,
                'input_tokens_cached': 2.0,
                'output_tokens': 6.0,
            },
            'codestral-2405': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.6,
            },
            'codestral-latest': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 0.6,
            },
            'pixtral-12b-2409': {
                'input_tokens_uncached': 0.15,
                'input_tokens_cached': 0.15,
                'output_tokens': 0.15,
            },
            'pixtral-12b': {
                'input_tokens_uncached': 0.15,
                'input_tokens_cached': 0.15,
                'output_tokens': 0.15,
            },
            'pixtral-12b-latest': {
                'input_tokens_uncached': 0.15,
                'input_tokens_cached': 0.15,
                'output_tokens': 0.15,
            },

            # OpenRouter

            'x-ai/grok-vision-beta': {
                'input_tokens_uncached': 5.0,
                'output_tokens': 15.0,
            },
            'meta-llama/llama-3.2-90b-vision-instruct': {
                'input_tokens_uncached': 0.8,
                'output_tokens': 1.6,
            },
            'google/gemini-pro-1.5': {
                'input_tokens_uncached': 1.25,
                'output_tokens': 5.0,
            },
            'google/gemini-2.0-flash-001': {
                'input_tokens_uncached': 0.1,
                'output_tokens': 0.4,
            },
            'qwen/qwen-2-vl-72b-instruct': {
                'input_tokens_uncached': 0.4,
                'output_tokens': 0.4,
            },
            'qwen/qwen-2-vl-7b-instruct': {
                'input_tokens_uncached': 0.1,
                'output_tokens': 0.1,
            },
            'x-ai/grok-2-vision-1212': {
                'input_tokens_uncached': 2.0,
                'output_tokens': 10.0,
            },
            'anthropic/claude-3-opus': {
                'input_tokens_uncached': 15.0,
                'output_tokens': 75.0,
            },
            'anthropic/claude-3.5-sonnet': {
                'input_tokens_uncached': 3.0,
                'output_tokens': 15.0,
            },
            'anthropic/claude-3-haiku': {
                'input_tokens_uncached': 0.25,
                'output_tokens': 1.25,
            },
            'minimax/minimax-01': {
                'input_tokens_uncached': 0.2,
                'output_tokens': 1.1,
            },

            # VLLM

            'nm-testing/pixtral-12b-FP8-dynamic': {
                'input_tokens_uncached': 0.0,
                'output_tokens': 0.0,
            }
        }

    def monitor_usage(self, msg):
        stamp = datetime.datetime.now().isoformat()

        self.get_logger().info(f"Registered '{msg.api_type}' usage - "
                               f"api_endpoint: '{msg.api_endpoint}', "
                               f"model_name: '{msg.model_name}', "
                               f"input_tokens_uncached: {msg.input_tokens_uncached}, "
                               f"input_tokens_cached: {msg.input_tokens_cached}, "
                               f"output_tokens: {msg.output_tokens}")

        api_types = ["completions", "embeddings"]
        if msg.api_type not in api_types:
            message = f"Ignoring usage-message of unsupported type '{msg.api_type}'. Supported usage types are {api_types}"
            self.get_logger().error(message)
            return

        if msg.api_endpoint == "":
            message = "Ignoring usage-message with empty field 'api_endpoint'"
            self.get_logger().error(message)
            return

        if msg.model_name == "":
            message = "Ignoring usage-message with empty field 'model_name'"
            self.get_logger().error(message)
            return

        if msg.input_tokens_uncached + msg.input_tokens_uncached + msg.input_tokens_uncached == 0:
            message = "Ignoring usage-message with zero token usage"
            self.get_logger().error(message)
            return

        self.file_lock.acquire()

        success, _, cache = self.read_usage()

        if not success:
            self.file_lock.release()
            self.get_logger().error("Ignoring registered usage after failure to read usage from cache")
            return

        if msg.api_type not in cache:
            cache[msg.api_type] = []

        cache_item = {
            'stamp': stamp,
            'api_endpoint': msg.api_endpoint,
            'model_name': msg.model_name,
        }

        if msg.input_tokens_uncached > 0:
            cache_item['input_tokens_uncached'] = msg.input_tokens_uncached
        if msg.input_tokens_cached > 0:
            cache_item['input_tokens_cached'] = msg.input_tokens_cached
        if msg.output_tokens > 0:
            cache_item['output_tokens'] = msg.output_tokens

        cache[msg.api_type].append(cache_item)

        self.cache_write_required = True

        self.write_usage(cache)

        self.file_lock.release()

    def get_usage(self, request, response):
        success = True

        api_types = ["", "completions", "embeddings"]
        if request.api_type not in api_types:
            success = False
            message = f"Unsupported usage type '{request.api_type}'. Supported usage types are {api_types}."
        else:
            if request.stamp_start == "":
                stamp_start = None
            else:
                try:
                    stamp_start = datetime.datetime.fromisoformat(request.stamp_start)
                except ValueError as e:
                    success = False
                    message = f"Failed to read field 'stamp_start': {e}"
            if request.stamp_end == "":
                stamp_end = None
            else:
                try:
                    stamp_end = datetime.datetime.fromisoformat(request.stamp_end)
                except ValueError as e:
                    success = False
                    message = f"Failed to read field 'stamp_end': {e}"

        if not success:
            self.get_logger().error(message)
            response.success = False
            response.message = message
            return response

        self.file_lock.acquire()

        success, message, cache = self.read_usage()

        self.file_lock.release()

        if not success:
            response.success = False
            response.message = message
            return response

        tic = time.perf_counter()

        usage = {}

        # TODO check cache content and throw error instead of crashing the node
        for api_type in cache:
            if request.api_type == api_type or request.api_type == "":

                if api_type == 'completions' or api_type == 'embeddings':
                    for item in cache[api_type]:

                        # for compatibility with old cache
                        if 'api_flavor' in item and 'api_endpoint' not in item:
                            item['api_endpoint'] = item['api_flavor']
                            del item['api_flavor']

                        if stamp_start is not None or stamp_end is not None:
                            stamp = datetime.datetime.fromisoformat(item['stamp'])
                        if stamp_start is not None:
                            if stamp < stamp_start:
                                continue
                        if stamp_end is not None:
                            if stamp > stamp_end:
                                continue

                        if api_type not in usage:
                            usage[api_type] = {}

                        if 'history' not in usage[api_type]:
                            usage[api_type]['history'] = []

                        usage[api_type]['history'].append(item)

                        if 'total' not in usage[api_type]:
                            usage[api_type]['total'] = {}
                        if item['api_endpoint'] not in usage[api_type]['total']:
                            usage[api_type]['total'][item['api_endpoint']] = {}
                        if item['model_name'] not in usage[api_type]['total'][item['api_endpoint']]:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']] = {}

                        input_tokens_uncached = usage[api_type]['total'][item['api_endpoint']][item['model_name']].get('input_tokens_uncached', 0) + item.get('input_tokens_uncached', 0)
                        if input_tokens_uncached > 0:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']]['input_tokens_uncached'] = input_tokens_uncached

                        input_tokens_cached = usage[api_type]['total'][item['api_endpoint']][item['model_name']].get('input_tokens_cached', 0) + item.get('input_tokens_cached', 0)
                        if input_tokens_cached > 0:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']]['input_tokens_cached'] = input_tokens_cached

                        output_tokens = usage[api_type]['total'][item['api_endpoint']][item['model_name']].get('output_tokens', 0) + item.get('output_tokens', 0)
                        if output_tokens > 0:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']]['output_tokens'] = output_tokens

        for api_type in api_types:
            if api_type in usage:
                total_dollars = 0.0
                if 'total' in usage[api_type]:
                    for api_endpoint in usage[api_type]['total']:
                        for model_name in usage[api_type]['total'][api_endpoint]:
                            if model_name in self.pricing:
                                input_tokens_uncached = usage[api_type]['total'][api_endpoint][model_name].get('input_tokens_uncached', 0)
                                input_tokens_cached = usage[api_type]['total'][api_endpoint][model_name].get('input_tokens_cached', 0)
                                output_tokens = usage[api_type]['total'][api_endpoint][model_name].get('output_tokens', 0)

                                if input_tokens_uncached > 0:
                                    if 'input_tokens_uncached' not in self.pricing[model_name]:
                                        self.get_logger().warn(f"Cannot consider price of '{input_tokens_uncached}' uncached prompt tokens for model '{model_name}'")
                                if input_tokens_cached > 0:
                                    if 'input_tokens_cached' not in self.pricing[model_name]:
                                        self.get_logger().warn(f"Cannot consider price of '{input_tokens_cached}' cached prompt tokens for model '{model_name}'")
                                if output_tokens > 0:
                                    if 'output_tokens' not in self.pricing[model_name]:
                                        self.get_logger().warn(f"Cannot consider price of '{output_tokens}' completion tokens for model '{model_name}'")

                                input_tokens_uncached_price = (input_tokens_uncached / 1000000) * self.pricing[model_name].get('input_tokens_uncached', 0.0)
                                input_tokens_cached_price = (input_tokens_cached / 1000000) * self.pricing[model_name].get('input_tokens_cached', 0.0)
                                output_tokens_price = (output_tokens / 1000000) * self.pricing[model_name].get('output_tokens', 0.0)

                                dollars = input_tokens_uncached_price + input_tokens_cached_price + output_tokens_price
                                total_dollars += dollars
                                usage[api_type]['total'][api_endpoint][model_name]['dollars'] = dollars
                            else:
                                self.get_logger().warn(f"Cannot estimate price for model '{model_name}'")
                usage[api_type]['total_dollars'] = total_dollars

        total_dollars = 0.0
        for api_type in usage:
            if 'total_dollars' in usage[api_type]:
                total_dollars += usage[api_type]['total_dollars']
        if len(usage) > 0:
            usage['total_dollars'] = total_dollars

        response.success = True

        if request.api_type != "":
            usage = usage.get(request.api_type, {})
        if ORJSON_AVAILABLE:
            response.usage = orjson.dumps(usage, option=orjson.OPT_INDENT_2).decode("utf-8")
        else:
            self.get_logger().warn("Using slow 'json' module to format usage. Install 'orjson' to speed this up!", once=True)
            response.usage = json.dumps(usage, indent=2)
        # self.get_logger().debug(f"Usage:\n{response.usage}")

        if request.api_type == "":
            response.message = f"Successfully retrieved usage in '{time.perf_counter() - tic:.3f}s'."
        else:
            response.message = f"Successfully retrieved '{request.api_type}' usage in '{time.perf_counter() - tic:.3f}s'."
        self.get_logger().debug(response.message)

        return response

    def read_usage(self):
        if self.cache_read_once is True and self.cache is not None:
            return True, None, self.cache

        cache_path = os.path.join(self.cache_folder, self.cache_file)

        try:
            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
                self.get_logger().debug(f"Created cache folder '{self.cache_folder}'")
            if not os.path.exists(cache_path):
                with open(cache_path, 'w') as f:
                    json.dump({}, f, indent=2)
                self.get_logger().info(f"Initialized usage cache file '{cache_path}'")
        except Exception as e:
            success = False
            message = f"Usage cache file does not exist but initializing it under '{cache_path}' failed: {e}"
            self.get_logger().error(message)
        else:
            success, message, cache = read_json(file_path=cache_path, logger=self.get_logger())
            if success:
                if not isinstance(cache, dict):
                    success = False
                    message = f"Expected content of usage cache file to be of type 'dict', but it is of type '{type(cache).__name__}'."
                    self.get_logger().error(message)

        if not success:
            cache = {}

        if self.cache_read_once:
            self.cache = cache

        return success, message, cache

    def write_usage(self, cache, force=False):
        if cache is None or not self.cache_write_required:
            return

        self.cache = cache

        if self.cache_write_lazy and not force:
            return

        cache_path = os.path.join(self.cache_folder, self.cache_file)
        success, _ = write_json(file_path=cache_path, json_object=cache, indent=True, logger=self.get_logger())
        if success:
            self.cache_write_required = False

    def write_cache_lazy(self):
        if self.cache_write_lazy:
            self.file_lock.acquire()
            self.write_usage(self.cache, force=True)
            self.file_lock.release()

def main(args=None):
    start_and_spin_node(UsageMonitor, args=args)

if __name__ == '__main__':
    main()
