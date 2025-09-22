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
from std_msgs.msg import String

from nimbro_api_interfaces.srv import UsageGet

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, read_json, write_json, assert_type_value, assert_keys, log_lines

### <Parameter Defaults>

node_name = "usage_monitor"
severity = 10

cache_read_once = True
cache_write_lazy = True
cache_write_interval = 30.0
cache_folder = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache")
cache_file = "cache_usage.json"

pricing_path = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "nimbro_api", "misc", "pricing.json")

### </Parameter Defaults>

class UsageMonitor(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)
        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

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
            name="cache_read_once",
            dtype=bool,
            default_value=cache_read_once,
            description="Read usage cache file once when required and keep it in memory instead of loading it every time.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_write_lazy",
            dtype=bool,
            default_value=cache_write_lazy,
            description="Write usage cache file in fixed intervals instead of writing it with every update.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_write_interval",
            dtype=float,
            default_value=cache_write_interval,
            description="Minimum time in seconds in which the usage cache file is written if cache_write_lazy is active.",
            read_only=True,
            range_min=10.0,
            range_max=3600.0,
            range_step=0.0
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

        self.parameter_handler.declare(
            name="pricing_path",
            dtype=str,
            default_value=pricing_path,
            description="Path to the pricing file that stores the model cost per 1M tokens. Set empty string to disable price calculation.",
            read_only=False
        )

        self.file_lock = threading.Lock()
        self.cache_write_required = False
        self.cache = None

        self.warned_model_missing = []

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_ALL, depth=100)
        self.sub_usage = self.create_subscription(
            msg_type=String,
            topic=f"{self.node_namespace}/api_usage".replace("//", "/"),
            callback=self.monitor_usage,
            qos_profile=qos_profile,
            callback_group=ReentrantCallbackGroup()
        )
        self.srv_get_usage = self.create_service(
            srv_type=UsageGet,
            srv_name=f"{self.node_namespace}/{self.node_name}/get_usage".replace("//", "/"),
            callback=self.get_usage,
            qos_profile=qos_profile,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.timer_write_lazy = self.create_timer(self.parameters.cache_write_interval, self.write_cache_lazy, callback_group=MutuallyExclusiveCallbackGroup())

        self._logger.info("Node started")

    def __del__(self):
        self.write_cache_lazy()
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self._logger.set_settings(settings={'severity': value})

        elif name == "cache_folder":
            if value == "":
                value = os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "cache")

        elif name == "pricing_path":
            # TODO structure this by API type and endpoint and retrieve costs for OpenRouter endpoint from their Models API
            if value == "":
                self.pricing = {}
            else:
                success, _message, pricing = read_json(file_path=value, logger=self._logger)
                if success:
                    if not isinstance(pricing, dict):
                        message = f"Expected content of pricing file to be of type 'dict' instead of '{type(pricing).__name__}'."
                        value = None
                    else:
                        self.pricing = pricing
                        self._logger.debug(f"Using pricing:\n{json.dumps(self.pricing, indent=4)}")

        return value, message

    def monitor_usage(self, msg):
        # log_lines(f"Received usage:\n{msg.data}", line_length=150, line_highlight="|", block_format=False, logger=self._logger, severity=10)

        try:
            usage = json.loads(msg.data)
        except Exception as e:
            message = f"Ignoring usage-message after failure to parse it as JSON: {e}"
            self._logger.error(message)
            return

        required_keys = ['api_type', 'api_endpoint', 'model_name', 'stamp_start', 'stamp_stop', 'duration']
        optional_keys = ['identifier', 'tokens_input_uncached', 'tokens_input_cached', 'tokens_output']

        try:
            usage = json.loads(msg.data)
            assert_type_value(usage, dict, name="usage message", logger=self._logger)
            assert_keys(obj=usage, keys=required_keys, mode="required", name="usage message", logger=self._logger)
            assert_keys(obj=usage, keys=required_keys + optional_keys, mode="whitelist", name="usage message", logger=self._logger)

            assert_type_value(usage['api_type'], ["completions", "embeddings"], name="field 'api_type' of usage message", logger=self._logger)

            assert_type_value(usage['api_endpoint'], str, name="field 'api_endpoint' of usage message", logger=self._logger)
            assert usage['api_endpoint'] != "", "Expected value of field 'api_endpoint' of usage message to not be an empty string."

            assert_type_value(usage['model_name'], str, name="field 'model_name' of usage message", logger=self._logger)
            assert usage['model_name'] != "", "Expected value of field 'model_name' of usage message to not be an empty string."

            assert_type_value(usage['stamp_start'], str, name="field 'stamp_start' of usage message", logger=self._logger)
            assert usage['stamp_start'] != "", "Expected value of field 'stamp_start' of usage message to not be an empty string." #

            assert_type_value(usage['model_name'], str, name="field 'model_name' of usage message", logger=self._logger)
            assert usage['model_name'] != "", "Expected value of field 'model_name' of usage message to not be an empty string."

            datetime.datetime.fromisoformat(usage['stamp_start'])
            datetime.datetime.fromisoformat(usage['stamp_stop'])

            assert_type_value(usage['duration'], [int, float], name="field 'duration' of usage message", logger=self._logger)

            if 'identifier' in usage:
                assert_type_value(usage['identifier'], str, name="optional field 'identifier' of usage message", logger=self._logger)
                assert usage['identifier'] != "", "Expected value of optional field 'identifier' of usage message to not be an empty string."

            if 'input_tokens_uncached' in usage:
                assert_type_value(usage['input_tokens_uncached'], int, name="optional field 'input_tokens_uncached' of usage message", logger=self._logger)
                assert usage['input_tokens_uncached'] > 0, "Expected value of optional field 'input_tokens_uncached' of usage message to be greater zero."

            if 'input_tokens_cached' in usage:
                assert_type_value(usage['input_tokens_cached'], int, name="optional field 'input_tokens_cached' of usage message", logger=self._logger)
                assert usage['input_tokens_cached'] > 0, "Expected value of optional field 'input_tokens_cached' of usage message to be greater zero."

            if 'tokens_output' in usage:
                assert_type_value(usage['tokens_output'], int, name="optional field 'tokens_output' of usage message", logger=self._logger)
                assert usage['tokens_output'] > 0, "Expected value of optional field 'tokens_output' of usage message to be greater zero."

        except Exception:
            self._logger.warn("Ignoring usage-message after failure to parse it")
            return

        log_lines(f"Registered usage:\n{msg.data}", line_length=150, line_highlight="|", block_format=False, logger=self._logger, severity=10)

        self.file_lock.acquire()

        success, _, cache = self.read_usage()
        if not success:
            self.file_lock.release()
            self._logger.error("Ignoring usage-message after failure to read usage from cache")
            return

        if usage['api_type'] not in cache:
            cache[usage['api_type']] = []
        cache[usage['api_type']].append(usage)

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
            self._logger.error(message)
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
                            if 'stamp_stop' in item:
                                stamp = datetime.datetime.fromisoformat(item['stamp_stop'])
                            else:
                                stamp = datetime.datetime.fromisoformat(item['stamp']) # for compatibility with old cache
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
                                        self._logger.warn(f"Cannot consider price of '{tokens_input_uncached}' uncached prompt tokens for model '{model_name}'")
                                if tokens_input_cached > 0:
                                    if 'tokens_input_cached' not in self.pricing[model_name]:
                                        self._logger.warn(f"Cannot consider price of '{tokens_input_cached}' cached prompt tokens for model '{model_name}'")
                                if tokens_output > 0:
                                    if 'tokens_output' not in self.pricing[model_name]:
                                        self._logger.warn(f"Cannot consider price of '{tokens_output}' completion tokens for model '{model_name}'")

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

                            elif model_name not in self.warned_model_missing:
                                self._logger.warn(f"Cannot estimate price for model '{model_name}'")
                                self.warned_model_missing.append(model_name)

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
            self._logger.warn("Using slow 'json' module to format usage. Install 'orjson' to speed this up!", once=True)
            response.usage = json.dumps(usage, indent=2)
        # self._logger.debug(f"Usage:\n{response.usage}")

        if request.api_type == "":
            response.message = f"Retrieved usage in '{time.perf_counter() - tic:.3f}s'."
        else:
            response.message = f"Retrieved '{request.api_type}' usage in '{time.perf_counter() - tic:.3f}s'."
        self._logger.debug(response.message)

        return response

    def read_usage(self):
        if self.parameters.cache_read_once is True and self.cache is not None:
            return True, None, self.cache

        cache_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)

        try:
            if not os.path.exists(self.parameters.cache_folder):
                os.makedirs(self.parameters.cache_folder)
                self._logger.debug(f"Created cache folder '{self.parameters.cache_folder}'")
            if not os.path.exists(cache_path):
                with open(cache_path, 'w') as f:
                    json.dump({}, f, indent=2)
                self._logger.info(f"Initialized usage cache file '{cache_path}'")
        except Exception as e:
            success = False
            message = f"Usage cache file does not exist but initializing it under '{cache_path}' failed: {repr(e)}"
            self._logger.error(message)
        else:
            success, message, cache = read_json(file_path=cache_path, logger=self._logger)
            if success:
                if not isinstance(cache, dict):
                    success = False
                    message = f"Expected content of usage cache file to be of type 'dict', but it is of type '{type(cache).__name__}'."
                    self._logger.error(message)

        if not success:
            cache = {}

        if self.parameters.cache_read_once:
            self.cache = cache

        return success, message, cache

    def write_usage(self, cache, force=False):
        if cache is None or not self.cache_write_required:
            return

        self.cache = cache

        if self.parameters.cache_write_lazy and not force:
            return

        cache_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)
        success, _ = write_json(file_path=cache_path, json_object=cache, indent=True, logger=self._logger)
        if success:
            self.cache_write_required = False

    def write_cache_lazy(self):
        if self.parameters.cache_write_lazy:
            self.file_lock.acquire()
            self.write_usage(self.cache, force=True)
            self.file_lock.release()

def main(args=None):
    start_and_spin_node(UsageMonitor, args=args)

if __name__ == '__main__':
    main()
