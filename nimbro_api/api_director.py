#!/usr/bin/env python3

import re
import json
import time
import random
import string
import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters as rcl_GetParameters

from nimbro_api.utils.node import block_until_future_complete, SelfShutdown
from nimbro_api_interfaces.srv import GetEmbeddings, GetImage, GetSpeech, GetUsage
from nimbro_api_interfaces.srv import CompletionsManage, CompletionsGetStatus, CompletionsGetSettings
from nimbro_api_interfaces.srv import CompletionsPrompt, CompletionsStop, CompletionsGetTools, CompletionsSetTools, CompletionsGetContext, CompletionsRemoveContext

# TODO add function documentation
# TODO make retry support int

class ApiDirector:

    def __init__(self, node,
                 completions_multiplexer_name="/nimbro_api/completions_multiplexer",
                 embeddings_name="/nimbro_api/embeddings",
                 images_name="/nimbro_api/images",
                 speech_name="/nimbro_api/speech",
                 usage_monitor_name="/nimbro_api/usage_monitor",
                 logger_name=None,
                 logger_severity=20):
        assert isinstance(node, rclpy.node.Node), f"ApiDirector requires a reference to an object of type 'rclpy.node.Node' but got a reference to an object of type '{type(node).__name__}'!"
        assert isinstance(completions_multiplexer_name, str), f"Expected 'completions_multiplexer_name' to be of type 'str' but it is of type '{type(completions_multiplexer_name).__name__}'!"
        assert isinstance(embeddings_name, str), f"Expected 'embeddings_name' to be of type 'str' but it is of type '{type(embeddings_name).__name__}'!"
        assert isinstance(images_name, str), f"Expected 'images_name' to be of type 'str' but it is of type '{type(images_name).__name__}'!"
        assert isinstance(speech_name, str), f"Expected 'speech_name' to be of type 'str' but it is of type '{type(speech_name).__name__}'!"
        assert isinstance(usage_monitor_name, str), f"Expected 'usage_monitor_name' to be of type 'str' but it is of type '{type(usage_monitor_name).__name__}'!"
        assert logger_name is None or isinstance(logger_name, str), f"Expected 'logger_name' to be of type 'None' or 'str' but it is of type '{type(logger_name).__name__}'!"
        assert logger_severity in [10, 20, 30, 40, 50], "Expected 'logger_severity' to be in [10, 20, 30, 40, 50] but it is not!"

        self._node = node

        if logger_name is None:
            logger_name = (f"{self._node.get_namespace()}.{self._node.get_name()}.api_director").replace("/", "")
            if logger_name[0] == ".":
                logger_name = logger_name[1:]
        self.set_logger(logger_name, logger_severity)

        completions_multiplexer_name = "/" + re.sub(r'^/+|/+$', '', completions_multiplexer_name)
        self._logger.debug(f"Using completions-multiplexer node '{completions_multiplexer_name}'")
        embeddings_name = "/" + re.sub(r'^/+|/+$', '', embeddings_name)
        self._logger.debug(f"Using embeddings node '{embeddings_name}'")
        images_name = "/" + re.sub(r'^/+|/+$', '', images_name)
        self._logger.debug(f"Using images node '{images_name}'")
        speech_name = "/" + re.sub(r'^/+|/+$', '', speech_name)
        self._logger.debug(f"Using speech node '{speech_name}'")
        usage_monitor_name = "/" + re.sub(r'^/+|/+$', '', usage_monitor_name)
        self._logger.debug(f"Using usage-monitor node '{usage_monitor_name}'")

        self._service_timeout = 7.0
        self._logger.debug(f"Initialized service timeout to '{self._service_timeout}s'")

        self._completion_timeout = 500.0 # TODO doesn't update properly and ugly because it is not individually targeting each node
        self._logger.debug(f"Initialized completion timeout to '{self._completion_timeout}s'")

        self._async_responses = {}

        cbg_services = MutuallyExclusiveCallbackGroup()

        self._cli_completions_prompt = self._node.create_client(CompletionsPrompt, completions_multiplexer_name + "/prompt", callback_group=cbg_services)
        self._cli_completions_stop = self._node.create_client(CompletionsStop, completions_multiplexer_name + "/stop", callback_group=MutuallyExclusiveCallbackGroup())
        self._cli_completions_get_tools = self._node.create_client(CompletionsGetTools, completions_multiplexer_name + "/get_tools", callback_group=cbg_services)
        self._cli_completions_set_tools = self._node.create_client(CompletionsSetTools, completions_multiplexer_name + "/set_tools", callback_group=cbg_services)
        self._cli_completions_get_context = self._node.create_client(CompletionsGetContext, completions_multiplexer_name + "/get_context", callback_group=MutuallyExclusiveCallbackGroup())
        self._cli_completions_remove_context = self._node.create_client(CompletionsRemoveContext, completions_multiplexer_name + "/remove_context", callback_group=cbg_services)

        self._cli_completions_manage = self._node.create_client(CompletionsManage, completions_multiplexer_name + "/manage", callback_group=cbg_services)
        self._cli_completions_get_status = self._node.create_client(CompletionsGetStatus, completions_multiplexer_name + "/get_status", callback_group=MutuallyExclusiveCallbackGroup())
        self._cli_completions_get_settings = self._node.create_client(CompletionsGetSettings, completions_multiplexer_name + "/get_settings", callback_group=cbg_services)

        self._cli_get_embeddings = self._node.create_client(GetEmbeddings, embeddings_name + "/get_embeddings", callback_group=MutuallyExclusiveCallbackGroup())
        self._cli_get_image = self._node.create_client(GetImage, images_name + "/get_image", callback_group=MutuallyExclusiveCallbackGroup())
        self._cli_get_speech = self._node.create_client(GetSpeech, speech_name + "/get_speech", callback_group=MutuallyExclusiveCallbackGroup())
        self._cli_get_usage = self._node.create_client(GetUsage, usage_monitor_name + "/get_usage", callback_group=MutuallyExclusiveCallbackGroup())

        # self._cli_get_timeout_parameters = self._node.create_client(rcl_GetParameters, completions_multiplexer_name + "/get_parameters", callback_group=MutuallyExclusiveCallbackGroup())
        # self._timer_timeout_params = self._node.create_timer(5.0, self._get_timeout_params, callback_group=MutuallyExclusiveCallbackGroup())

    def _log_return(self, completions_id, success, message, *args):
        if completions_id is None:
            completions_prefix = ""
        elif isinstance(completions_id, str):
            completions_prefix = "[" + completions_id + "] "
        else:
            completions_prefix = "[INVALID_COMPLETIONS_ID] "

        if success:
            if message is None:
                pass
            elif message == "":
                self._logger.warn(completions_prefix + "Function successfully terminated with empty message")
            else:
                self._logger.info(completions_prefix + message)
        else:
            if message is None:
                pass
            elif message == "":
                self._logger.error(completions_prefix + "Function failed with empty message")
            else:
                self._logger.error(completions_prefix + message)

        return (success, message) + args

    def _get_timeout_params(self):
        # self._timer_timeout_params._Timer__timer.change_timer_period(int(10 * 1e9)) # not available in foxy

        if self._cli_get_timeout_parameters.service_is_ready():
            try:
                request = rcl_GetParameters.Request()
                request.names.append("timeout_service")
                request.names.append("timeout_completion")
                future = self._cli_get_timeout_parameters.call_async(request)
                block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                if future.done():
                    response = future.result()
                    if response.values[0].type == ParameterType.PARAMETER_NOT_SET:
                        pass
                        # self._logger.error(f"Failed to retrieve timeout parameter '{request.names[0]}' because it is not set.")
                    else:
                        if response.values[0].double_value != self._service_timeout:
                            self._service_timeout = response.values[0].double_value
                            # self._logger.info(f"Set service timeout to '{self._service_timeout}s'")
                    if response.values[1].type == ParameterType.PARAMETER_NOT_SET:
                        pass
                        # self._logger.error(f"Failed to retrieve timeout parameter '{request.names[1]}' because it is not set.")
                    else:
                        if response.values[1].double_value != self._completion_timeout:
                            self._completion_timeout = response.values[1].double_value
                            # self._logger.info(f"Set completion timeout to '{self._completion_timeout}s'")
                else:
                    self._cli_get_timeout_parameters.remove_pending_request(future)
                    # self._logger.error(f"Cannot retrieve timeout parameters because the service '{self._cli_get_timeout_parameters.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s').")
            except Exception as e:
                e
                # self._logger.error(f"Failed to retrieve timeout parameters ({e}).")
            except KeyboardInterrupt:
                raise SelfShutdown
        else:
            pass
            # self._logger.warn(f"Cannot retrieve timeout parameters because the service '{self._cli_get_timeout_parameters.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s').", once=True)

    def _get_async_id(self, completions_id):
        while True:
            a, b = self._node.get_clock().now().seconds_nanoseconds()
            c = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(8))
            async_id = f"{a}_{b}_{c}"
            if async_id in self._async_responses:
                self._logger.warn(f"[{completions_id}] Failed to generate unique asynchronous ID.")
            else:
                return async_id

    def _async_thread(self, async_id, args):
        if self._async_responses[async_id]['succeed_async_id'] is not None:
            # wait until thread previous is done
            while 'response' not in self._async_responses[self._async_responses[async_id]['succeed_async_id']]:
                self._logger.info(f"[{self._async_responses[async_id]['completions_id']}] Asynchronous thread '{async_id}' waiting for termination of previous asynchronous thread '{self._async_responses[async_id]['succeed_async_id']}'.", throttle_duration_sec=1.0, skip_first=True)
                time.sleep(0.1)
            # if previous thread failed, cancel this thread
            if self._async_responses[self._async_responses[async_id]['succeed_async_id']]['response'][0] is False:
                cancel = True
                # TODO some ignore_success field where succession is required but success is irrelevant?
                if cancel:
                    message = f"Asynchronous thread '{async_id}' not forwarded because previous asynchronous thread '{self._async_responses[async_id]['succeed_async_id']}' did not succeed."
                    self._async_responses[async_id]['response'] = (False, message, None, None)
                    self._logger.info(f"[{self._async_responses[async_id]['completions_id']}] {message}")
                    self._async_responses[async_id]['terminated'] = self._node.get_clock().now()

        if 'response' not in self._async_responses[async_id]:
            self._async_responses[async_id]['started'] = self._node.get_clock().now()
            if self._async_responses[async_id]['succeed_async_id'] is not None:
                time_waited = (self._async_responses[async_id]['started'] - self._async_responses[async_id]['registered']).nanoseconds / 1e9
                self._logger.info(f"[{self._async_responses[async_id]['completions_id']}] Asynchronous thread '{async_id}' started after waiting '{time_waited:.3f}s'.")
            if self._async_responses[async_id]['type'] == "prompt":
                self._async_responses[async_id]['response'] = self.prompt(*args)
            else:
                self._async_responses[async_id]['response'] = self.set_tools(*args)

            self._async_responses[async_id]['terminated'] = self._node.get_clock().now()
            time_waited = (self._async_responses[async_id]['terminated'] - self._async_responses[async_id]['started']).nanoseconds / 1e9
            self._logger.info(f"[{self._async_responses[async_id]['completions_id']}] Asynchronous thread '{async_id}' terminated after '{time_waited:.3f}s'.")

    def async_status(self):
        if len(self._async_responses) == 0:
            self._logger.info("There were no asynchronous threads registered yet.")
        else:
            self._logger.info("Asynchronous thread info:")
            for result in self._async_responses:
                self._logger.info(f"ID '{result}': {self._async_responses[result]}")

    def set_logger(self, name=None, severity=None):
        if name is not None:
            assert isinstance(name, str), f"Expected 'name' to be of type 'str' but it is of type '{type(name).__name__}'!"
            self._logger = rclpy.logging.get_logger(name)
            self._logger.debug(f"Set logger name to '{name}'")

        if severity is not None:
            assert severity in [10, 20, 30, 40, 50], "Expected 'severity' to be in [10, 20, 30, 40, 50] but it is not!"
            # rclpy.logging.set_logger_level(f"{self._node.get_namespace()}/{name}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(severity))
            rclpy.logging.set_logger_level(self._logger.name, rclpy.logging.LoggingSeverity(severity))
            self._logger.debug(f"Set logger severity to '{severity}'")

    ## Chat Completions API

    # Completions Allocation

    def get_status(self, retry=False):
        if not isinstance(retry, bool):
            return self._log_return(None, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None, None)

        completions_ids = None
        acquired = None

        while True:
            try:
                available = self._cli_completions_get_status.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot retrieve completions status because the service '{self._cli_completions_get_status.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsGetStatus.Request()
                        future = self._cli_completions_get_status.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            if success:
                                completions_ids = response.completions_id
                                acquired = response.acquired
                        else:
                            self._cli_completions_get_status.remove_pending_request(future)
                            success = False
                            message = f"Cannot retrieve completions status because the service '{self._cli_completions_get_status.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to retrieve completions status ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn("Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        if message == "":
            message = None
        return self._log_return(None, success, message, completions_ids, acquired)

    # TODO add reset_tools
    def acquire(self, reset_parameters=True, reset_context=True, retry=False):
        if not isinstance(retry, bool):
            return self._log_return(None, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        completions_id = None

        while True:
            try:
                available = self._cli_completions_manage.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    completions_id = None
                    success = False
                    message = f"Cannot acquire completions because the service '{self._cli_completions_manage.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsManage.Request()
                        request.completions_id = ""
                        request.action = "acquire"
                        request.parameter_names = []
                        request.parameter_values = []
                        future = self._cli_completions_manage.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            completions_id = None if response.completions_id == "" else response.completions_id
                            success = response.success
                            message = response.message
                        else:
                            self._cli_completions_manage.remove_pending_request(future)
                            completions_id = None
                            success = False
                            message = f"Cannot acquire completions because the service '{self._cli_completions_manage.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        completions_id = None
                        success = False
                        message = f"Failed to acquire completions ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Failed to acquire completions, retrying until success...", throttle_duration_sec=self._service_timeout)

        if success and (reset_parameters or reset_context):
            if reset_parameters:
                success_params, message_params = self.reset_parameters(completions_id, retry=retry)
                success = success and success_params
                message = (message + " " + message_params).strip()
            if reset_context:
                success_msg, message_msg = self.remove_context(completions_id, remove_all=True, retry=retry)
                success = success and success_msg
                message = (message + " " + message_msg).strip()

        return self._log_return(completions_id, success, message, completions_id)

    def duplicate(self, completions_id, retry=False):
        success, message, new_completions_id = self.acquire(reset_parameters=True, reset_context=True, retry=retry)
        if success:
            success, message, parameters = self.get_parameters(completions_id=completions_id, retry=retry)
            if success:
                success, message = self.set_parameters(completions_id=new_completions_id, parameter_names=list(parameters.keys()), parameter_values=[str(value) for value in list(parameters.values())], retry=retry)
                if success:
                    success, message, tools = self.get_tools(completions_id=completions_id, retry=retry)
                    if success:
                        success, message = self.set_tools(completions_id=new_completions_id, tools=tools, retry=retry)
                        if success:
                            success, message, messages = self.get_context(completions_id=completions_id, retry=retry)
                            if success:
                                for message in messages:
                                    success, message, text_response, tool_calls = self.prompt(completions_id=new_completions_id, text=message, role='json', reset_context=False, tool_response_id="", response_type="none", retry=retry)
                                    if not success:
                                        break
        if success:
            message = f"Duplicated completions '{completions_id}' to new completions '{new_completions_id}'."
        else:
            message = f"Failed to duplicate completions '{completions_id}' to new completions '{new_completions_id}' ({message[:-1]})."

        return self._log_return(new_completions_id, success, message, new_completions_id)

    def release(self, completions_id, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.")
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.")

        while True:
            try:
                available = self._cli_completions_manage.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot release completions because the service '{self._cli_completions_manage.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsManage.Request()
                        request.completions_id = completions_id
                        request.action = "release"
                        request.parameter_names = []
                        request.parameter_values = []
                        future = self._cli_completions_manage.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                        else:
                            self._cli_completions_manage.remove_pending_request(future)
                            success = False
                            message = f"Cannot release completions because the service '{self._cli_completions_manage.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to release completions ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message)

    # Model Parameters

    def set_parameters(self, completions_id, parameter_names=[], parameter_values=[], retry=False):
        # parameter_names: [logger_level, probe_api_connection, api_flavor, model_name, model_temperatur, model_top_p, model_max_tokens, model_presence_penalty, model_frequency_penalty,
        #                   normalize_text_response, max_tool_calls_per_response, correction_attempts, timeout_chunk, timeout_completion]

        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.")

        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        elif isinstance(parameter_names, list):
            for i in range(len(parameter_names)):
                if not isinstance(parameter_names[i], str):
                    return self._log_return(completions_id, False, f"Provided argument 'parameter_names' contains element of invalid type '{type(parameter_names[i]).__name__}'. Supported type is 'str'.")
        else:
            return self._log_return(completions_id, False, f"Provided argument 'parameter_names' is of invalid type '{type(parameter_names).__name__}'. Supported types are 'str' and 'list'.")

        if isinstance(parameter_values, str):
            parameter_values = [parameter_values]
        elif isinstance(parameter_values, list):
            for i in range(len(parameter_values)):
                if not isinstance(parameter_values[i], str):
                    return self._log_return(completions_id, False, f"Provided argument 'parameter_values' contains element of invalid type '{type(parameter_values[i]).__name__}'. Supported type is 'str' (values will be converted from string as required by the parameter).")
        else:
            return self._log_return(completions_id, False, f"Provided argument 'parameter_values' is of invalid type '{type(parameter_values).__name__}'. Supported types are 'str' (value will be converted from string as required by the parameter) and 'list'.")

        if not len(parameter_values) == len(parameter_values):
            return self._log_return(completions_id, False, f"The number of provided 'parameter_names' ({len(parameter_names)}) and 'parameter_values' ({len(parameter_values)}) do not match.")

        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.")

        while True:
            try:
                available = self._cli_completions_manage.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot set parameter '{parameter_names}' because the service '{self._cli_completions_manage.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsManage.Request()
                        request.completions_id = completions_id
                        request.action = "configure"
                        request.parameter_names = parameter_names
                        request.parameter_values = parameter_values
                        future = self._cli_completions_manage.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                        else:
                            self._cli_completions_manage.remove_pending_request(future)
                            success = False
                            message = f"Cannot set parameter '{parameter_names}' because the service '{self._cli_completions_manage.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to set parameter '{parameter_names}' ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message)

    def get_parameters(self, completions_id, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        parameters = None

        while True:
            try:
                available = self._cli_completions_get_settings.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot retrieve parameters because the service '{self._cli_completions_get_settings.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    request = CompletionsGetSettings.Request()
                    request.completions_id = completions_id
                    future = self._cli_completions_get_settings.call_async(request)
                    try:
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            try:
                                if len(response.parameter_names) != len(response.parameter_values) or len(response.parameter_names) != len(response.parameter_types):
                                    raise Exception(f"Expected number of parameter names '{len(response.parameter_names)}', values '{len(response.parameter_values)}', and types '{len(response.parameter_types)}' to match")
                                if success:
                                    parameters = {}
                                    for i in range(len(response.parameter_names)):
                                        if response.parameter_types[i] == ParameterType.PARAMETER_BOOL:
                                            parameters[response.parameter_names[i]] = bool(response.parameter_values[i])
                                        elif response.parameter_types[i] == ParameterType.PARAMETER_INTEGER:
                                            parameters[response.parameter_names[i]] = int(response.parameter_values[i])
                                        elif response.parameter_types[i] == ParameterType.PARAMETER_DOUBLE:
                                            parameters[response.parameter_names[i]] = float(response.parameter_values[i])
                                        elif response.parameter_types[i] == ParameterType.PARAMETER_STRING:
                                            parameters[response.parameter_names[i]] = response.parameter_values[i]
                                        else:
                                            raise Exception(f"Parameter type '{response.parameter_types[i]}' not implemented")
                            except Exception as e:
                                parameters = None
                                success = False
                                message = f"Failed to parse parameters ({e})."
                        else:
                            self._cli_completions_get_settings.remove_pending_request(future)
                            success = False
                            message = f"Cannot retrieve parameters because the service '{self._cli_completions_get_settings.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to retrieve parameters ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message, parameters)

    def reset_parameters(self, completions_id, retry=False):
        return self.set_parameters(completions_id=completions_id, parameter_names=[], parameter_values=[], retry=retry)

    # Prompting

    def async_prompt(self, completions_id, text, role="user", reset_context=False, tool_response_id=None, response_type="auto", retry=False, succeed_async_id=None):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None)
        if not isinstance(text, str) and not isinstance(text, dict):
            return self._log_return(completions_id, False, f"Provided argument 'text' is of invalid type '{type(text).__name__}'. Supported types are 'str' and 'dict'.", None)
        if isinstance(text, dict):
            try:
                text = json.dumps(text)
            except Exception as e:
                return self._log_return(completions_id, False, f"Failed to parse argument 'text' of type 'dict' as JSON ({e}).", None)
        if not isinstance(role, str):
            return self._log_return(completions_id, False, f"Provided argument 'role' is of invalid type '{type(role).__name__}'. Supported type is 'str'.", None)
        if not isinstance(reset_context, bool):
            return self._log_return(completions_id, False, f"Provided argument 'reset_context' is of invalid type '{type(reset_context).__name__}'. Supported type is 'bool'.", None)
        if tool_response_id is not None and not isinstance(tool_response_id, str):
            return self._log_return(completions_id, False, f"Provided argument 'tool_response_id' is of invalid type '{type(tool_response_id).__name__}'. Supported types are 'None' and 'str'.", None)
        if isinstance(tool_response_id, str):
            if tool_response_id == "":
                tool_response_id = None
        if not isinstance(response_type, str):
            return self._log_return(completions_id, False, f"Provided argument 'response_type' is of invalid type '{type(response_type).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)
        if not (succeed_async_id is None or isinstance(succeed_async_id, str)):
            return self._log_return(completions_id, False, f"Provided argument 'succeed_async_id' is of invalid type '{type(succeed_async_id).__name__}'. Supported types are 'None' and 'str'.", None)
        if succeed_async_id is not None and len(self._async_responses) == 0:
            return self._log_return(completions_id, False, f"Provided argument 'succeed_async_id' with invalid value '{succeed_async_id}'. Valid value is 'None'.", None)
        if succeed_async_id is not None and succeed_async_id not in self._async_responses.keys():
            return self._log_return(completions_id, False, f"Provided argument 'succeed_async_id' with invalid value '{succeed_async_id}'. Valid values are 'None' and elements of {list(self._async_responses.keys())}.", None)

        async_id = self._get_async_id(completions_id)

        self._async_responses[async_id] = {}
        self._async_responses[async_id]['type'] = "prompt"
        self._async_responses[async_id]['completions_id'] = completions_id
        self._async_responses[async_id]['succeed_async_id'] = succeed_async_id
        self._async_responses[async_id]['registered'] = self._node.get_clock().now()
        self._async_responses[async_id]['thread'] = threading.Thread(target=self._async_thread, args=(async_id, (completions_id, text, role, reset_context, tool_response_id, response_type, retry)))
        self._async_responses[async_id]['thread'].start()

        return self._log_return(completions_id, True, f"Registered asynchronous thread '{async_id}'.", async_id)

    def async_get(self, async_id, mute_timeout_logging=False, timeout=None):
        if not isinstance(async_id, str):
            return self._log_return(None, False, f"Provided argument 'async_id' is of invalid type '{type(async_id).__name__}'. Supported type is 'str'.", None)
        if async_id not in self._async_responses:
            return self._log_return(None, False, f"Cannot retrieve asynchronous thread response for ID '{async_id}' that does not exist.", None)
        if not isinstance(mute_timeout_logging, bool):
            return self._log_return(None, False, f"Provided argument 'mute_timeout_logging' is of invalid type '{type(mute_timeout_logging).__name__}'. Supported type is 'bool'.", None)
        if not (timeout is None or isinstance(timeout, int) or isinstance(timeout, float)):
            return self._log_return(None, False, f"Provided argument 'timeout' is of invalid type '{type(timeout).__name__}'. Supported types are 'None', int, and 'float'.", None)
        if timeout is not None:
            timeout = abs(timeout)

        now = self._node.get_clock().now()
        if self._async_responses[async_id]['thread'].is_alive():
            before = now
            while True:
                self._async_responses[async_id]['thread'].join(timeout=1.0 if timeout is None else min(timeout, 1.0))
                now = self._node.get_clock().now()
                time_waited = (now - before).nanoseconds / 1e9
                if timeout is not None:
                    if time_waited > timeout:
                        if mute_timeout_logging:
                            return False, f"Failed to receive response before timeout after '{time_waited:.3f}s'.", None
                        else:
                            return self._log_return(self._async_responses[async_id]['completions_id'], False, f"Failed to receive response before timeout after '{time_waited:.3f}s'.", None)
                if self._async_responses[async_id]['thread'].is_alive():
                    self._logger.info(f"[{self._async_responses[async_id]['completions_id']}] Waiting for response from asynchronous thread '{async_id}' since '{time_waited:.3f}s'.", throttle_duration_sec=1.0)
                else:
                    message = f"Retrieved response from asynchronous thread '{async_id}' after waiting '{time_waited:.3f}s'."
                    break
        else:
            message = f"Retrieved response from asynchronous thread '{async_id}' without waiting."

        if 'received' not in self._async_responses[async_id]:
            self._async_responses[async_id]['received'] = now

        return self._log_return(self._async_responses[async_id]['completions_id'], True, message, self._async_responses[async_id]['response'])

    def async_cancel(self, async_id):
        raise Exception("Not implemented") # TODO?
        # if response was received and successful remove it # depending on response_type 1 or 2 messages have to be removed
        # else if response was received and not successful return
        # else
        # send stop
        # wait for response
        # if response was successful remove it
        # return
        # How does this behave with queuing and retry?

    def async_set_tools(self, completions_id, tools, retry=False, succeed_async_id=None):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None)
        if not isinstance(tools, list):
            return self._log_return(completions_id, False, f"Provided argument 'tools' is of invalid type '{type(tools).__name__}'. Supported type is 'list'.", None)
        for i in range(len(tools)):
            if isinstance(tools[i], dict):
                try:
                    tools[i] = json.dumps(tools[i])
                except Exception as e:
                    return self._log_return(completions_id, False, f"Failed to parse dictionary '{tools[i]}' as string ({e}).", None)
            elif not isinstance(tools[i], str):
                return self._log_return(completions_id, False, f"Provided argument 'tools' contains element '{tools[i]}' of invalid type '{type(tools[i]).__name__}'. Supported types are 'str' and 'dict'.", None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        if not (succeed_async_id is None or isinstance(succeed_async_id, str)):
            return self._log_return(completions_id, False, f"Provided argument 'succeed_async_id' is of invalid type '{type(succeed_async_id).__name__}'. Supported types are 'None' and 'str'.", None)
        if succeed_async_id is not None and len(self._async_responses) == 0:
            return self._log_return(completions_id, False, f"Provided argument 'succeed_async_id' with invalid value '{succeed_async_id}'. Valid value is 'None'.", None)
        if succeed_async_id is not None and succeed_async_id not in self._async_responses.keys():
            return self._log_return(completions_id, False, f"Provided argument 'succeed_async_id' with invalid value '{succeed_async_id}'. Valid values are 'None' and elements of {list(self._async_responses.keys())}.", None)

        async_id = self._get_async_id(completions_id)

        self._async_responses[async_id] = {}
        self._async_responses[async_id]['type'] = "tools"
        self._async_responses[async_id]['completions_id'] = completions_id
        self._async_responses[async_id]['succeed_async_id'] = succeed_async_id
        self._async_responses[async_id]['registered'] = self._node.get_clock().now()
        self._async_responses[async_id]['thread'] = threading.Thread(target=self._async_thread, args=(async_id, (completions_id, tools, retry)))
        self._async_responses[async_id]['thread'].start()

        return self._log_return(completions_id, True, f"Registered asynchronous thread '{async_id}'.", async_id)

    def prompt(self, completions_id, text, role="user", reset_context=False, tool_response_id=None, response_type="auto", retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(text, str) and not isinstance(text, dict):
            return self._log_return(completions_id, False, f"Provided argument 'text' is of invalid type '{type(text).__name__}'. Supported types are 'str' and 'dict'.", None, None)
        if isinstance(text, dict):
            try:
                text = json.dumps(text)
            except Exception as e:
                return self._log_return(completions_id, False, f"Failed to parse argument 'text' of type 'dict' as JSON ({e}).", None, None)
        if not isinstance(role, str):
            return self._log_return(completions_id, False, f"Provided argument 'role' is of invalid type '{type(role).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(reset_context, bool):
            return self._log_return(completions_id, False, f"Provided argument 'reset_context' is of invalid type '{type(reset_context).__name__}'. Supported type is 'bool'.", None, None)
        if tool_response_id is not None and not isinstance(tool_response_id, str):
            return self._log_return(completions_id, False, f"Provided argument 'tool_response_id' is of invalid type '{type(tool_response_id).__name__}'. Supported types are 'None' and 'str'.", None, None)
        if isinstance(tool_response_id, str):
            if tool_response_id == "":
                tool_response_id = None
        if not isinstance(response_type, str):
            return self._log_return(completions_id, False, f"Provided argument 'response_type' is of invalid type '{type(response_type).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None, None)

        text_reply = None
        tool_calls = None
        # received_response = False

        while True:
            try:
                available = self._cli_completions_prompt.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot prompt completions because the service '{self._cli_completions_prompt.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    self._logger.info(f"[{completions_id}] Prompt: '{text}' (role:'{role}', reset:'{reset_context}', tool:'{tool_response_id}', type:'{response_type}')")
                    try:
                        request = CompletionsPrompt.Request()
                        request.completions_id = completions_id
                        request.role = role
                        request.text = text
                        request.reset_context = reset_context
                        if tool_response_id is not None:
                            request.tool_response_id = tool_response_id
                        else:
                            request.tool_response_id = ""
                        request.response_type = response_type
                        future = self._cli_completions_prompt.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._completion_timeout + self._service_timeout)
                        if future.done():
                            # received_response = True
                            response = future.result()
                            success = response.success

                            if response.text != '':
                                text_reply = response.text

                                if response_type == "json":
                                    try:
                                        text_reply = json.loads(response.text)
                                    except Exception as e:
                                        text_reply = response.text
                                        json_message = f"Failed to parse text response in JSON mode as dictionary ({e})."
                                        self._logger.error(f"[{completions_id}] " + json_message, throttle_duration_sec=self._service_timeout)

                            if len(response.tool_calls) > 0:
                                tool_calls = response.tool_calls
                                try:
                                    tool_calls = [json.loads(tool_call) for tool_call in tool_calls]
                                    if not isinstance(tool_calls, list):
                                        raise Exception(f"Tool calls are of type '{type(tool_calls).__name__}' instead of 'list'")
                                except Exception as e:
                                    tool_calls = None
                                    json_message = f"Failed to parse tool_calls as dictionary ({e})."
                                    self._logger.error(f"[{completions_id}] " + json_message, throttle_duration_sec=self._service_timeout)
                                else:
                                    for i, _ in enumerate(tool_calls):
                                        try:
                                            if not isinstance(tool_calls[i], dict):
                                                raise Exception(f"Tool call is of type '{type(tool_calls[i]).__name__}' instead of 'dict'")
                                            if "id" not in tool_calls[i]:
                                                raise Exception("Tool call is missing 'id' field")
                                            if not isinstance(tool_calls[i]['id'], str):
                                                raise Exception(f"Tool call 'id' is of type '{type(tool_calls[i]['id']).__name__}' instead of 'str'")
                                            if "name" not in tool_calls[i]:
                                                raise Exception("Tool call is missing 'name' field")
                                            if not isinstance(tool_calls[i]['name'], str):
                                                raise Exception(f"Tool call 'name' is of type '{type(tool_calls[i]['name']).__name__}' instead of 'str'")
                                            if "arguments" not in tool_calls[i]:
                                                raise Exception("Tool call is missing 'arguments' field")
                                            tool_calls[i]['arguments'] = json.loads(tool_calls[i]['arguments'])
                                            if not isinstance(tool_calls[i]["arguments"], dict):
                                                raise Exception(f"Tool call 'arguments' are of type '{type(tool_calls[i]['arguments']).__name__}' instead of 'dict'")
                                        except Exception as e:
                                            tool_calls[i] = None
                                            json_message = f"Failed to parse tool_call as dictionary ({e})."
                                            self._logger.error(f"[{completions_id}] " + json_message, throttle_duration_sec=self._service_timeout)

                                    tool_calls = [tool_call for tool_call in tool_calls if tool_call is not None]
                                    if len(tool_calls) == 0:
                                        tool_calls = None

                            if text_reply is None and tool_calls is not None:
                                message = f"Response: '{tool_calls}' (success={success}, msg='{response.message}', text=None)"
                            elif text_reply is not None:
                                message = f"Response: '{text_reply}' (success={success}, msg='{response.message}', tools=" + ('None' if tool_calls is None else f"'{tool_calls}'") + ")"
                            else:
                                message = f"Response: '{response.message}' (success={success}, text=" + ('None' if text_reply is None else f"'{text_reply}'") + ", tools=" + ('None' if tool_calls is None else f"'{tool_calls}'") + ")"
                        else:
                            self._cli_completions_prompt.remove_pending_request(future)
                            success = False
                            message = f"Cannot prompt completions because the service '{self._cli_completions_prompt.srv_name}' does not respond (Timeout after '{self._completion_timeout + self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to prompt completions ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                # if received_response or not retry:
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message, text_reply, tool_calls)

    def stop(self, completions_id, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        while True:
            try:
                available = self._cli_completions_stop.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot stop prompt because the service '{self._cli_completions_stop.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsStop.Request()
                        request.completions_id = completions_id
                        future = self._cli_completions_stop.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                        else:
                            self._cli_completions_stop.remove_pending_request(future)
                            success = False
                            message = f"Cannot stop prompt because the service '{self._cli_completions_stop.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to stop prompt ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message)

    def set_tools(self, completions_id, tools, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.")
        if not (tools is None or isinstance(tools, list)):
            return self._log_return(completions_id, False, f"Provided argument 'tools' is of invalid type '{type(tools).__name__}'. Supported types are 'None' and 'list'.")
        if tools is None:
            tools = []
        for i in range(len(tools)):
            if isinstance(tools[i], dict):
                try:
                    tools[i] = json.dumps(tools[i])
                except Exception as e:
                    return self._log_return(completions_id, False, f"Failed to parse dictionary '{tools[i]}' as string ({e}).")
            elif not isinstance(tools[i], str):
                return self._log_return(completions_id, False, f"Provided argument 'tools' contains element '{tools[i]}' of invalid type '{type(tools[i]).__name__}'. Supported types are 'str' and 'dict'.")
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.")

        while True:
            try:
                available = self._cli_completions_set_tools.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot set tools because the service '{self._cli_completions_set_tools.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    if len(tools) > 0:
                        tools_text = ""
                        for i in range(len(tools)):
                            tools_text += "\n" + tools[i]
                        self._logger.info(f"[{completions_id}] Set tools: {tools_text}")
                    try:
                        request = CompletionsSetTools.Request()
                        request.completions_id = completions_id
                        request.tools = tools
                        future = self._cli_completions_set_tools.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                        else:
                            self._cli_completions_set_tools.remove_pending_request(future)
                            success = False
                            message = f"Cannot set tools because the service '{self._cli_completions_set_tools.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to set tools ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message)

    def get_tools(self, completions_id, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        tools = None

        while True:
            try:
                available = self._cli_completions_get_tools.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot retrieve tools because the service '{self._cli_completions_get_tools.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsGetTools.Request()
                        request.completions_id = completions_id
                        future = self._cli_completions_get_tools.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            if success and len(response.tools) > 0:
                                tools = []
                                try:
                                    for i in range(len(response.tools)):
                                        tools.append(json.loads(response.tools[i]))
                                except Exception as e:
                                    success = False
                                    message = f"Failed to parse tool call '{response.tools[i]}' as dictionary ({e})."
                                    tools = None
                        else:
                            self._cli_completions_get_tools.remove_pending_request(future)
                            success = False
                            message = f"Cannot retrieve tools because the service '{self._cli_completions_get_tools.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to retrieve tools ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message, tools)

    # Message History

    def get_context(self, completions_id, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        context = None

        while True:
            try:
                available = self._cli_completions_get_context.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot retrieve message history because the service '{self._cli_completions_get_context.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsGetContext.Request()
                        request.completions_id = completions_id
                        future = self._cli_completions_get_context.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            if success:
                                context = []
                                try:
                                    for msg in response.context:
                                        context.append(json.loads(msg))
                                except Exception as e:
                                    context = None
                                    success = False
                                    message = f"Failed to parse message history ({e})."
                        else:
                            self._cli_completions_get_context.remove_pending_request(future)
                            success = False
                            message = f"Cannot retrieve message history because the service '{self._cli_completions_get_context.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to retrieve the message history ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message, context)

    def remove_context(self, completions_id, remove_all=False, message_index=0, indexing_last_to_first=True, retry=False):
        if not isinstance(completions_id, str):
            return self._log_return(None, False, f"Provided argument 'completions_id' is of invalid type '{type(completions_id).__name__}'. Supported type is 'str'.")
        if not isinstance(remove_all, bool):
            return self._log_return(completions_id, False, f"Provided argument 'remove_all' is of invalid type '{type(remove_all).__name__}'. Supported type is 'bool'.")
        if not isinstance(message_index, int):
            return self._log_return(completions_id, False, f"Provided argument 'message_index' is of invalid type '{type(message_index).__name__}'. Supported type is 'int'.")
        if not message_index >= 0:
            return self._log_return(completions_id, False, f"Provided argument 'message_index' is '{message_index}' but must greater or equal zero. You might want to use the 'indexing_last_to_first' argument.")
        if not isinstance(indexing_last_to_first, bool):
            return self._log_return(completions_id, False, f"Provided argument 'indexing_last_to_first' is of invalid type '{type(indexing_last_to_first).__name__}'. Supported type is 'bool'.")
        if not isinstance(retry, bool):
            return self._log_return(completions_id, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.")

        while True:
            try:
                available = self._cli_completions_remove_context.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot remove message '{message_index}'" + (" (indexing from last to first)" if indexing_last_to_first else "") + f" because the service '{self._cli_completions_remove_context.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = CompletionsRemoveContext.Request()
                        request.completions_id = completions_id
                        request.remove_all = remove_all
                        request.indexing_last_to_first = indexing_last_to_first
                        request.index = message_index
                        future = self._cli_completions_remove_context.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                        else:
                            self._cli_completions_remove_context.remove_pending_request(future)
                            success = False
                            if remove_all:
                                message = f"Cannot remove message '{message_index}'" + (" (indexing from last to first)" if indexing_last_to_first else "") + f" because the service '{self._cli_completions_remove_context.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                            else:
                                message = f"Cannot remove message all because the service '{self._cli_completions_remove_context.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        if remove_all:
                            message = f"Failed to remove all messages ({e})."
                        else:
                            message = f"Failed to remove message '{message_index}'" + (" (indexing from last to first)" if indexing_last_to_first else "") + f" ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{completions_id}] " + message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn(f"[{completions_id}] Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(completions_id, success, message)

    # Other APIs

    def get_embeddings(self, text, retry=False):
        if not (isinstance(text, str) or isinstance(text, list)):
            return self._log_return(None, False, f"Provided argument 'text' is of invalid type '{type(text).__name__}'. Supported types are 'str' and 'list'.", None)
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text
        for t in text_list:
            if not isinstance(t, str):
                return self._log_return(None, False, f"Provided list 'text' features element of invalid type '{type(t).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(None, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        embeddings = None

        while True:
            try:
                available = self._cli_get_embeddings.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot get embeddings because the service '{self._cli_get_embeddings.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = GetEmbeddings.Request()
                        request.texts = text_list
                        future = self._cli_get_embeddings.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._completion_timeout + self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            embeddings = [list(embedding.embedding) for embedding in response.embeddings]
                        else:
                            self._cli_get_embeddings.remove_pending_request(future)
                            success = False
                            message = f"Cannot get embeddings because the service '{self._cli_get_embeddings.srv_name}' does not respond (Timeout after '{self._completion_timeout + self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to get embeddings ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn("Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(None, success, message, embeddings[0] if success and isinstance(text, str) else embeddings)

    def get_image(self, prompt, model="", quality="", style="", size="", retry=False):
        if not isinstance(prompt, str):
            return self._log_return(None, False, f"Provided argument 'prompt' is of invalid type '{type(prompt).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(model, str):
            return self._log_return(None, False, f"Provided argument 'model' is of invalid type '{type(model).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(quality, str):
            return self._log_return(None, False, f"Provided argument 'quality' is of invalid type '{type(quality).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(size, str):
            return self._log_return(None, False, f"Provided argument 'size' is of invalid type '{type(size).__name__}'. Supported type is 'str'.", None, None)
        if not isinstance(retry, bool):
            return self._log_return(None, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None, None)

        image, path = None, None

        while True:
            try:
                available = self._cli_get_image.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot get image because the service '{self._cli_get_image.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = GetImage.Request()
                        request.prompt = prompt
                        request.model = model
                        request.quality = quality
                        request.style = style
                        request.size = size
                        future = self._cli_get_image.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._completion_timeout + self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            image = response.image
                            path = response.path
                            # TODO optionally convert image to numpy
                        else:
                            self._cli_get_image.remove_pending_request(future)
                            success = False
                            message = f"Cannot get image because the service '{self._cli_get_image.srv_name}' does not respond (Timeout after '{self._completion_timeout + self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to get image ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn("Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(None, success, message, image, path)

    def get_speech(self, text, model="", voice="", speed=1.0, retry=False):
        if not isinstance(text, str):
            return self._log_return(None, False, f"Provided argument 'text' is of invalid type '{type(text).__name__}'. Supported type is 'str'.", None)
        if not isinstance(model, str):
            return self._log_return(None, False, f"Provided argument 'model' is of invalid type '{type(model).__name__}'. Supported type is 'str'.", None)
        if not isinstance(voice, str):
            return self._log_return(None, False, f"Provided argument 'voice' is of invalid type '{type(voice).__name__}'. Supported type is 'str'.", None)
        if not (isinstance(speed, float) or isinstance(speed, int)):
            return self._log_return(None, False, f"Provided argument 'speed' is of invalid type '{type(speed).__name__}'. Supported types are 'float' and 'int'.", None)
        speed = float(speed)
        if not isinstance(retry, bool):
            return self._log_return(None, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        path = None

        while True:
            try:
                available = self._cli_get_speech.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot get speech because the service '{self._cli_get_speech.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = GetSpeech.Request()
                        request.text = text
                        request.model = model
                        request.voice = voice
                        request.speed = speed
                        future = self._cli_get_speech.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._completion_timeout + self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            path = response.path
                            # TODO read file and forward as numpy
                        else:
                            self._cli_get_speech.remove_pending_request(future)
                            success = False
                            message = f"Cannot get speech because the service '{self._cli_get_speech.srv_name}' does not respond (Timeout after '{self._completion_timeout + self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to get speech ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn("Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(None, success, message, path)

    # Usage

    def get_usage(self, api_type="", stamp_start="", stamp_end="", retry=False):
        if not isinstance(api_type, str):
            return self._log_return(None, False, f"Provided argument 'api_type' is of invalid type '{type(api_type).__name__}'. Supported type is 'str'.", None)
        if not isinstance(stamp_start, str):
            return self._log_return(None, False, f"Provided argument 'stamp_start' is of invalid type '{type(stamp_start).__name__}'. Supported type is 'str'.", None)
        if not isinstance(stamp_end, str):
            return self._log_return(None, False, f"Provided argument 'stamp_end' is of invalid type '{type(stamp_end).__name__}'. Supported type is 'str'.", None)
        if not isinstance(retry, bool):
            return self._log_return(None, False, f"Provided argument 'retry' is of invalid type '{type(retry).__name__}'. Supported type is 'bool'.", None)

        usage = None

        while True:
            try:
                available = self._cli_get_usage.wait_for_service(timeout_sec=self._service_timeout)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Cannot retrieve usage because the service '{self._cli_get_usage.srv_name}' is not available (Timeout after '{self._service_timeout:.3f}s')."
                else:
                    try:
                        request = GetUsage.Request()
                        request.api_type = api_type
                        request.stamp_start = stamp_start
                        request.stamp_end = stamp_end
                        future = self._cli_get_usage.call_async(request)
                        block_until_future_complete(self._node, future, timeout_sec=self._service_timeout)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                            if success:
                                try:
                                    usage = json.loads(response.usage)
                                except Exception as e:
                                    usage = None
                                    success = False
                                    message = f"Failed to parse usage ({e})."
                        else:
                            self._cli_get_usage.remove_pending_request(future)
                            success = False
                            message = f"Cannot retrieve usage because the service '{self._cli_get_usage.srv_name}' does not respond (Timeout after '{self._service_timeout:.3f}s')."
                    except Exception as e:
                        success = False
                        message = f"Failed to retrieve usage ({e})."
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or not retry:
                    break
                else:
                    if message != "":
                        self._logger.warn(message, throttle_duration_sec=self._service_timeout)
                    self._logger.warn("Response was not successful, retrying until success...", throttle_duration_sec=self._service_timeout)

        return self._log_return(None, success, message, usage)
