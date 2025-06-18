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

pricing_path = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "pricing.json")

### </Parameter Defaults>

class UsageMonitor(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)
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

        descriptor = ParameterDescriptor()
        descriptor.name = "pricing_path"
        descriptor.type = ParameterType.PARAMETER_STRING
        descriptor.description = "Path to the pricing file that stores the model cost per 1M tokens. Set empty string to disable price calculation."
        descriptor.read_only = False
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, pricing_path, descriptor)

        self.parameter_handler.all_declared()

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

        elif parameter.name == "pricing_path":
            # TODO structure this by API type and endpoint and retrieve costs for OpenRouter endpoint from their Models API instead
            if parameter.value == "":
                self.pricing = {}
            else:
                success, reason, pricing = read_json(file_path=parameter.value, logger=self.get_logger())
                if success:
                    if not isinstance(pricing, dict):
                        success = False
                        reason = f"Expected content of pricing file to be of type 'dict' instead of '{type(pricing).__name__}'."
                    else:
                        self.pricing = pricing
                        self.get_logger().debug(f"Using pricing:\n{json.dumps(self.pricing, indent=4)}")
        else:
            return None, None

        return success, reason

    def monitor_usage(self, msg):
        stamp = datetime.datetime.now().isoformat()

        self.get_logger().info(f"Registered '{msg.api_type}' usage - "
                               f"api_type: '{msg.api_type}', "
                               f"api_endpoint: '{msg.api_endpoint}', "
                               f"model_name: '{msg.model_name}', "
                               f"identifier: '{msg.identifier}', "
                               f"tokens_input_uncached: {msg.tokens_input_uncached}, "
                               f"tokens_input_cached: {msg.tokens_input_cached}, "
                               f"tokens_output: {msg.tokens_output}")

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

        if msg.tokens_input_uncached + msg.tokens_input_uncached + msg.tokens_input_uncached == 0:
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

        if msg.identifier != "":
            cache_item['identifier'] = msg.identifier
        if msg.tokens_input_uncached > 0:
            cache_item['tokens_input_uncached'] = msg.tokens_input_uncached
        if msg.tokens_input_cached > 0:
            cache_item['tokens_input_cached'] = msg.tokens_input_cached
        if msg.tokens_output > 0:
            cache_item['tokens_output'] = msg.tokens_output

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
            if request.api_endpoint == "":
                filter_api_endpoint = None
            else:
                filter_api_endpoint = request.api_endpoint
            if request.model_name == "":
                filter_model_name = None
            else:
                filter_model_name = request.model_name
            if request.identifier == "":
                filter_identifier = None
            else:
                filter_identifier = request.identifier
            if request.stamp_start == "":
                filter_stamp_start = None
            else:
                try:
                    filter_stamp_start = datetime.datetime.fromisoformat(request.stamp_start)
                except ValueError as e:
                    success = False
                    message = f"Failed to read field 'stamp_start': {repr(e)}"
            if request.stamp_end == "":
                filter_stamp_end = None
            else:
                try:
                    filter_stamp_end = datetime.datetime.fromisoformat(request.stamp_end)
                except ValueError as e:
                    success = False
                    message = f"Failed to read field 'stamp_end': {repr(e)}"

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
            if request.api_type == "" or request.api_type == api_type:
                if api_type == 'completions' or api_type == 'embeddings':
                    for item in cache[api_type]:

                        # for compatibility with old cache
                        if 'api_flavor' in item and 'api_endpoint' not in item:
                            item['api_endpoint'] = item['api_flavor']
                            del item['api_flavor']
                        if 'input_tokens_uncached' in item:
                            item['tokens_input_uncached'] = item['input_tokens_uncached']
                            del item['input_tokens_uncached']
                        if 'input_tokens_cached' in item:
                            item['tokens_input_cached'] = item['input_tokens_cached']
                            del item['input_tokens_cached']
                        if 'output_tokens' in item:
                            item['tokens_output'] = item['output_tokens']
                            del item['output_tokens']

                        # filters
                        if filter_api_endpoint is not None:
                            if item.get('api_endpoint') != filter_api_endpoint:
                                continue
                        if filter_model_name is not None:
                            if item.get('model_name') != filter_model_name:
                                continue
                        if filter_identifier is not None:
                            if item.get('identifier') != filter_identifier:
                                continue
                        if filter_stamp_start is not None or filter_stamp_end is not None:
                            stamp = datetime.datetime.fromisoformat(item['stamp'])
                        if filter_stamp_start is not None:
                            if stamp < filter_stamp_start:
                                continue
                        if filter_stamp_end is not None:
                            if stamp > filter_stamp_end:
                                continue

                        # dollars per item
                        if item.get('model_name') in self.pricing:
                            tokens_input_uncached_price = (item.get('tokens_input_uncached', 0.0) / 1000000) * self.pricing[item['model_name']].get('tokens_input_uncached', 0.0)
                            tokens_input_cached_price = (item.get('tokens_input_cached', 0.0) / 1000000) * self.pricing[item['model_name']].get('tokens_input_cached', 0.0)
                            tokens_output_price = (item.get('tokens_output', 0.0) / 1000000) * self.pricing[item['model_name']].get('tokens_output', 0.0)
                            if tokens_input_uncached_price > 0 or tokens_input_cached_price > 0:
                                item['dollars_input'] = tokens_input_uncached_price + tokens_input_cached_price
                            if tokens_output_price > 0:
                                item['dollars_output'] = tokens_output_price
                            if 'dollars_input' in item and 'dollars_output' in item:
                                item['dollars_total'] = tokens_input_uncached_price + tokens_input_cached_price + tokens_output_price

                        # history
                        if api_type not in usage:
                            usage[api_type] = {}
                        if 'history' not in usage[api_type]:
                            usage[api_type]['history'] = []
                        usage[api_type]['history'].append(item)

                        # usage per api_type
                        if 'total' not in usage[api_type]:
                            usage[api_type]['total'] = {}
                        if item['api_endpoint'] not in usage[api_type]['total']:
                            usage[api_type]['total'][item['api_endpoint']] = {}
                        if item['model_name'] not in usage[api_type]['total'][item['api_endpoint']]:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']] = {}

                        tokens_input_uncached = usage[api_type]['total'][item['api_endpoint']][item['model_name']].get('tokens_input_uncached', 0) + item.get('tokens_input_uncached', 0)
                        if tokens_input_uncached > 0:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']]['tokens_input_uncached'] = tokens_input_uncached

                        tokens_input_cached = usage[api_type]['total'][item['api_endpoint']][item['model_name']].get('tokens_input_cached', 0) + item.get('tokens_input_cached', 0)
                        if tokens_input_cached > 0:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']]['tokens_input_cached'] = tokens_input_cached

                        tokens_output = usage[api_type]['total'][item['api_endpoint']][item['model_name']].get('tokens_output', 0) + item.get('tokens_output', 0)
                        if tokens_output > 0:
                            usage[api_type]['total'][item['api_endpoint']][item['model_name']]['tokens_output'] = tokens_output

        # dollars per api_endpoint and api_type
        for api_type in api_types:
            if api_type in usage:
                total_dollars_input = 0.0
                total_dollars_output = 0.0
                if 'total' in usage[api_type]:
                    for api_endpoint in usage[api_type]['total']:
                        for model_name in usage[api_type]['total'][api_endpoint]:
                            if model_name in self.pricing:
                                tokens_input_uncached = usage[api_type]['total'][api_endpoint][model_name].get('tokens_input_uncached', 0)
                                tokens_input_cached = usage[api_type]['total'][api_endpoint][model_name].get('tokens_input_cached', 0)
                                tokens_output = usage[api_type]['total'][api_endpoint][model_name].get('tokens_output', 0)

                                if tokens_input_uncached > 0:
                                    if 'tokens_input_uncached' not in self.pricing[model_name]:
                                        self.get_logger().warn(f"Cannot consider price of '{tokens_input_uncached}' uncached prompt tokens for model '{model_name}'")
                                if tokens_input_cached > 0:
                                    if 'tokens_input_cached' not in self.pricing[model_name]:
                                        self.get_logger().warn(f"Cannot consider price of '{tokens_input_cached}' cached prompt tokens for model '{model_name}'")
                                if tokens_output > 0:
                                    if 'tokens_output' not in self.pricing[model_name]:
                                        self.get_logger().warn(f"Cannot consider price of '{tokens_output}' completion tokens for model '{model_name}'")

                                tokens_input_uncached_price = (tokens_input_uncached / 1000000) * self.pricing[model_name].get('tokens_input_uncached', 0.0)
                                tokens_input_cached_price = (tokens_input_cached / 1000000) * self.pricing[model_name].get('tokens_input_cached', 0.0)
                                tokens_output_price = (tokens_output / 1000000) * self.pricing[model_name].get('tokens_output', 0.0)

                                dollars_input = tokens_input_uncached_price + tokens_input_cached_price
                                if dollars_input > 0:
                                    total_dollars_input += dollars_input
                                    usage[api_type]['total'][api_endpoint][model_name]['dollars_input'] = dollars_input

                                if tokens_output_price > 0:
                                    total_dollars_output += tokens_output_price
                                    usage[api_type]['total'][api_endpoint][model_name]['dollars_output'] = tokens_output_price

                                if dollars_input > 0 and tokens_output_price > 0:
                                    usage[api_type]['total'][api_endpoint][model_name]['dollars_total'] = dollars_input + tokens_output_price

                            else:
                                self.get_logger().warn(f"Cannot estimate price for model '{model_name}'")

                if total_dollars_input > 0:
                    usage[api_type]['dollars_input'] = total_dollars_input
                if total_dollars_input > 0:
                    usage[api_type]['dollars_output'] = total_dollars_output
                if total_dollars_input > 0 and total_dollars_output > 0:
                    usage[api_type]['dollars_total'] = total_dollars_input + total_dollars_output

        # dollars across all api_type
        total_dollars_input = 0.0
        total_dollars_output = 0.0
        for api_type in usage:
            if 'dollars_input' in usage[api_type]:
                total_dollars_input += usage[api_type]['dollars_input']
            if 'dollars_output' in usage[api_type]:
                total_dollars_output += usage[api_type]['dollars_output']
        if total_dollars_input > 0:
            usage['dollars_input'] = total_dollars_input
        if total_dollars_output > 0:
            usage['dollars_output'] = total_dollars_output
        if total_dollars_input > 0 and total_dollars_output > 0:
            usage['dollars_total'] = total_dollars_input + total_dollars_output

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
            message = f"Usage cache file does not exist but initializing it under '{cache_path}' failed: {repr(e)}"
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
