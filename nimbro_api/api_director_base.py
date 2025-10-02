#!/usr/bin/env python3

import re
import copy
import json
import time
import random
import string
import datetime
import threading
import traceback

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import builtin_interfaces.msg

from nimbro_api_interfaces.srv import NimbroVisionGet, EmbeddingsGet, ImagesGet, SpeechGet, UsageGet
from nimbro_api_interfaces.srv import CompletionsManage, CompletionsStatusGet, CompletionsSettingsGet
from nimbro_api_interfaces.srv import CompletionsPrompt, CompletionsInterrupt, CompletionsToolsGet, CompletionsToolsSet, CompletionsContextGet, CompletionsContextSet

from nimbro_utils.lazy import Logger, SelfShutdown, block_until_future_complete, assert_type_value, assert_keys, assert_log, update_dict, convert_stamp, read_json

class ApiDirectorBase:

    def __init__(self, node, settings=None):
        # node
        assert_type_value(
            obj=node,
            type_or_value=rclpy.node.Node,
            name="argument 'node'"
        )
        self._node = node

        # logger:
        self._logger = Logger(self._node, settings={
            'severity': settings['severity'],
            'prefix': None,
            'name': settings['suffix']
        })

        # settings
        self._set_settings(settings=settings, keep_existing=False)

    # Internals

    def _client_wrapper(self, prefix, client, request, timeout_service, timeout_response, retry):
        assert_type_value(obj=retry, type_or_value=[int, bool], name="argument 'retry'", logger=self._logger)
        if isinstance(retry, bool):
            if retry:
                retry = -1
            else:
                retry = 0

        response = None

        while True:
            try:
                available = client.wait_for_service(timeout_sec=timeout_service)
            except KeyboardInterrupt:
                raise SelfShutdown
            else:
                if not available:
                    success = False
                    message = f"Failed to find service '{client.srv_name}': Timeout after '{timeout_service:.3f}s'."
                else:
                    try:
                        future = client.call_async(request)
                        block_until_future_complete(self._node, future, timeout=timeout_response)
                        if future.done():
                            response = future.result()
                            success = response.success
                            message = response.message
                        else:
                            client.remove_pending_request(future)
                            success = False
                            message = f"Failed to obtain response from service '{client.srv_name}': Timeout after '{timeout_response:.3f}s'."
                    except Exception as e:
                        self._logger.error(f"{traceback.format_exc()}")
                        success = False
                        message = f"An unexpected error occurred: {repr(e)}"
                    except KeyboardInterrupt:
                        raise SelfShutdown
                if success or retry == 0:
                    break
                else:
                    if message != "":
                        self._logger.warn(f"[{prefix}] {message}", throttle_duration_sec=timeout_service)
                    if retry == -1:
                        self._logger.warn(f"[{prefix}] Retrying until success...", throttle_duration_sec=timeout_service)
                    else:
                        self._logger.warn(f"[{prefix}] Retrying '{retry}' more time{'' if retry == 1 else 's'}...", throttle_duration_sec=timeout_service)
                        retry -= 1

        return success, message, response

    def _log_return(self, prefix, success, message, *args):
        if success:
            if message == "":
                self._logger.warn(f"[{prefix}] Function terminated with empty message.")
            else:
                self._logger.info(f"[{prefix}] {message}")
        else:
            if message == "":
                self._logger.error(f"[{prefix}] Function failed with empty message.")
            else:
                self._logger.error(f"[{prefix}] {message}")

        return (success, message) + args

    def _get_async_id(self, completions_id):
        while True:
            a, b = self._node.get_clock().now().seconds_nanoseconds()
            c = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(8))
            async_id = f"{a}_{b}_{c}"
            if async_id in self._async_responses:
                self._logger.warn("Failed to generate unique asynchronous ID.")
            else:
                return async_id

    def _async_thread(self, async_id, args):
        if self._async_responses[async_id]['succeed_async_id'] is not None:
            # wait until thread previous is done
            while 'response' not in self._async_responses[self._async_responses[async_id]['succeed_async_id']]:
                self._logger.debug(f"[{self._async_responses[async_id]['completions_id']}.async_thread] Asynchronous thread '{async_id}' waiting for termination of preceding asynchronous thread '{self._async_responses[async_id]['succeed_async_id']}'.", throttle_duration_sec=1.0, skip_first=True)
                time.sleep(0.01)
            # if previous thread failed, cancel this thread
            if self._async_responses[self._async_responses[async_id]['succeed_async_id']]['response'][0] is False:
                cancel = True
                # TODO some ignore_success field where succession is required but success is irrelevant?
                if cancel:
                    message = f"Asynchronous thread '{async_id}' not forwarded because preceding asynchronous thread '{self._async_responses[async_id]['succeed_async_id']}' failed." # TODO forward message of failed thread
                    self._async_responses[async_id]['response'] = (False, message, None)
                    self._logger.warn(f"[{self._async_responses[async_id]['completions_id']}.async_thread] {message}")
                    self._async_responses[async_id]['terminated'] = self._node.get_clock().now()

        if 'response' not in self._async_responses[async_id]:
            self._async_responses[async_id]['started'] = self._node.get_clock().now()
            if self._async_responses[async_id]['succeed_async_id'] is not None:
                time_waited = (self._async_responses[async_id]['started'] - self._async_responses[async_id]['registered']).nanoseconds / 1e9
                self._logger.debug(f"[{self._async_responses[async_id]['completions_id']}.async_thread] Asynchronous thread '{async_id}' started after waiting '{time_waited:.3f}s'.")
            if self._async_responses[async_id]['type'] == "prompt":
                self._async_responses[async_id]['response'] = self.prompt(*args)
            elif self._async_responses[async_id]['type'] == "tools":
                self._async_responses[async_id]['response'] = self.set_tools(*args)
            else:
                self._async_responses[async_id]['response'] = self.set_parameters(*args)

            self._async_responses[async_id]['terminated'] = self._node.get_clock().now()
            time_waited = (self._async_responses[async_id]['terminated'] - self._async_responses[async_id]['started']).nanoseconds / 1e9
            self._logger.debug(f"[{self._async_responses[async_id]['completions_id']}.async_thread] Asynchronous thread '{async_id}' terminated after '{time_waited:.3f}s'.")

    # ApiDirectorBase Settings

    def _get_settings(self):
        """
        Retrieve the current settings of the ApiDirectorBase.

        Returns:
            dict: A deep copy of the current settings.
        """
        return copy.deepcopy(self._settings)

    def _set_settings(self, settings, keep_existing):
        """
        Update settings of the ApiDirector.

        Args:
            settings (dict): New settings to apply.
            keep_existing (bool, optional): If True, merge with existing settings. Otherwise, replace current settings entirely. Defaults to True.

        Raises:
            AssertionError: If input arguments or provided settings are invalid.
        """
        # parse arguments
        assert_type_value(obj=keep_existing, type_or_value=bool, name="argument 'keep_existing'", logger=self._logger)
        settings = update_dict(old_dict=self._settings if keep_existing else {}, new_dict=settings, key_name="setting", logger=self._logger, info=False, debug=False)
        default_settings_names = [
            'severity', 'suffix', 'timeout_service', 'timeout_response', 'node_completions_multiplexer',
            'node_embeddings', 'node_images', 'node_speech', 'node_nimbro_vision', 'node_usage_monitor', 'voice_presets'
        ]
        assert_keys(obj=settings, keys=default_settings_names, mode="match", name="settings", logger=self._logger)

        # Logger
        self._logger.set_settings({'severity': settings['severity'], 'name': settings['suffix']})

        # node names
        create_client = {}
        for name in ['node_completions_multiplexer', 'node_embeddings', 'node_images', 'node_speech', 'node_nimbro_vision', 'node_usage_monitor']:
            assert_type_value(obj=settings[name], type_or_value=str, name=f"setting '{name}'", logger=self._logger)
            settings[name] = "/" + re.sub(r'^/+|/+$', '', settings[name])
            create_client[name] = True

        # timeouts
        for name in ['timeout_service', 'timeout_response']:
            assert_type_value(obj=settings[name], type_or_value=[float, int], name=f"setting '{name}'", logger=self._logger)
            if settings[name] < 0.0:
                message = f"Expected settings '{name}' to be larger or equal zero but got '{settings[name]}'."
                self._logger.error(message)
                assert settings[name] > 0.0, message

        # read voice presets
        if isinstance(settings['voice_presets'], str):
            success, message, settings['voice_presets'] = read_json(file_path=settings['voice_presets'], name="file", logger=self._logger)
            assert success, message
        assert_type_value(obj=settings['voice_presets'], type_or_value=dict, name="setting 'voice_presets'", logger=self._logger)
        for key in settings['voice_presets']:
            assert_type_value(obj=key, type_or_value=str, name="all keys in setting 'voice_presets'", logger=self._logger)
            assert_type_value(obj=settings['voice_presets'][key], type_or_value=dict, name="all values in setting 'voice_presets'", logger=self._logger)
            assert_keys(obj=settings['voice_presets'][key], keys=['voice', 'instructions'], mode="match", name="all values in setting 'voice_presets'", logger=self._logger)
            assert_type_value(obj=settings['voice_presets'][key]['voice'], type_or_value=str, name="key 'voice' in setting 'voice_presets'", logger=self._logger)
            assert_type_value(obj=settings['voice_presets'][key]['instructions'], type_or_value=str, name="key 'instructions' in setting 'voice_presets'", logger=self._logger)

        # create / update interfaces

        if hasattr(self, "_settings"):
            if settings['node_completions_multiplexer'] == self._settings['node_completions_multiplexer']:
                create_client['node_completions_multiplexer'] = False
            else:
                self._node.destroy_client(self._cli_completions_prompt)
                self._node.destroy_client(self._cli_completions_interrupt)
                self._node.destroy_client(self._cli_completions_get_tools)
                self._node.destroy_client(self._cli_completions_set_tools)
                self._node.destroy_client(self._cli_completions_get_context)
                self._node.destroy_client(self._cli_completions_set_context)
                self._node.destroy_client(self._cli_completions_manage)
                self._node.destroy_client(self._cli_completions_get_status)
                self._node.destroy_client(self._cli_completions_get_settings)

            if settings['node_embeddings'] == self._settings['node_embeddings']:
                create_client['node_embeddings'] = False
            else:
                self._node.destroy_client(self._cli_get_embeddings)

            if settings['node_images'] == self._settings['node_images']:
                create_client['node_images'] = False
            else:
                self._node.destroy_client(self._cli_get_images)

            if settings['node_speech'] == self._settings['node_speech']:
                create_client['node_speech'] = False
            else:
                self._node.destroy_client(self._cli_get_speech)

            if settings['node_nimbro_vision'] == self._settings['node_nimbro_vision']:
                create_client['node_nimbro_vision'] = False
            else:
                self._node.destroy_client(self._cli_mmgroundingdino)
                self._node.destroy_client(self._cli_sam2_realtime_update)
                self._node.destroy_client(self._cli_sam2_realtime_track)
                self._node.destroy_client(self._cli_dam)
                self._node.destroy_client(self._cli_kosmos2)
                self._node.destroy_client(self._cli_florence2)

            if settings['node_usage_monitor'] == self._settings['node_usage_monitor']:
                create_client['node_usage_monitor'] = False
            else:
                self._node.destroy_client(self._cli_get_usage)
        else:
            self._qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=50)
            self._async_responses = {}

        if create_client['node_completions_multiplexer']:
            self._cli_completions_prompt = self._node.create_client(CompletionsPrompt, f"{settings['node_completions_multiplexer']}/prompt", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_interrupt = self._node.create_client(CompletionsInterrupt, f"{settings['node_completions_multiplexer']}/interrupt", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_get_tools = self._node.create_client(CompletionsToolsGet, f"{settings['node_completions_multiplexer']}/get_tools", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_set_tools = self._node.create_client(CompletionsToolsSet, f"{settings['node_completions_multiplexer']}/set_tools", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_get_context = self._node.create_client(CompletionsContextGet, f"{settings['node_completions_multiplexer']}/get_context", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_set_context = self._node.create_client(CompletionsContextSet, f"{settings['node_completions_multiplexer']}/set_context", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_manage = self._node.create_client(CompletionsManage, f"{settings['node_completions_multiplexer']}/manage", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_get_status = self._node.create_client(CompletionsStatusGet, f"{settings['node_completions_multiplexer']}/get_status", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            self._cli_completions_get_settings = self._node.create_client(CompletionsSettingsGet, f"{settings['node_completions_multiplexer']}/get_settings", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        if create_client['node_embeddings']:
            self._cli_get_embeddings = self._node.create_client(EmbeddingsGet, f"{settings['node_embeddings']}/get_embeddings", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        if create_client['node_images']:
            self._cli_get_images = self._node.create_client(ImagesGet, f"{settings['node_images']}/get_image", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        if create_client['node_speech']:
            self._cli_get_speech = self._node.create_client(SpeechGet, f"{settings['node_speech']}/get_speech", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        if create_client['node_nimbro_vision']:
            self._cli_mmgroundingdino = self._node.create_client(NimbroVisionGet, f"{settings['node_nimbro_vision']}/mmgroundingdino", qos_profile=self._qos_profile, callback_group=ReentrantCallbackGroup())
            self._cli_sam2_realtime_update = self._node.create_client(NimbroVisionGet, f"{settings['node_nimbro_vision']}/sam2_realtime_update", qos_profile=self._qos_profile, callback_group=ReentrantCallbackGroup())
            self._cli_sam2_realtime_track = self._node.create_client(NimbroVisionGet, f"{settings['node_nimbro_vision']}/sam2_realtime_track", qos_profile=self._qos_profile, callback_group=ReentrantCallbackGroup())
            self._cli_dam = self._node.create_client(NimbroVisionGet, f"{settings['node_nimbro_vision']}/dam", qos_profile=self._qos_profile, callback_group=ReentrantCallbackGroup())
            self._cli_kosmos2 = self._node.create_client(NimbroVisionGet, f"{settings['node_nimbro_vision']}/kosmos2", qos_profile=self._qos_profile, callback_group=ReentrantCallbackGroup())
            self._cli_florence2 = self._node.create_client(NimbroVisionGet, f"{settings['node_nimbro_vision']}/florence2", qos_profile=self._qos_profile, callback_group=ReentrantCallbackGroup())

        if create_client['node_usage_monitor']:
            self._cli_get_usage = self._node.create_client(UsageGet, f"{settings['node_usage_monitor']}/get_usage", qos_profile=self._qos_profile, callback_group=MutuallyExclusiveCallbackGroup())

        self._settings = settings

    # Chat Completions API - Management

    def _get_status(self, retry):
        prefix = "completions.get_status"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_get_status,
            request=CompletionsStatusGet.Request(),
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        if success:
            completions_ids = response.completions_id
            acquired = response.acquired
        else:
            completions_ids = None
            acquired = None

        return self._log_return(prefix, success, message, completions_ids, acquired)

    def _acquire(self, reset_parameters, reset_context, retry):
        assert_type_value(obj=reset_parameters, type_or_value=bool, name="argument 'reset_parameters'", logger=self._logger)
        assert_type_value(obj=reset_context, type_or_value=bool, name="argument 'reset_context'", logger=self._logger)

        request = CompletionsManage.Request()
        request.completions_id = ""
        request.action = "acquire"
        request.parameter_names = []
        request.parameter_values = []

        prefix = "completions.acquire"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_manage,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        if success:
            completions_id = response.completions_id
            self._logger.info(f"[{completions_id}.acquire] {message}")
            if reset_parameters:
                success_params, message_params = self.reset_parameters(completions_id, retry=retry)
                success = success and success_params
                message = (message + " " + message_params).strip()
            if reset_context:
                success_msg, message_msg = self.set_context(completions_id, mode="reset", new_messages=[], retry=retry)
                success = success and success_msg
                message = (message + " " + message_msg).strip()
            return success, message, completions_id
        else:
            return self._log_return(prefix, False, message, None)

    def _duplicate(self, completions_id, retry):
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
                            success, message, context = self.get_context(completions_id=completions_id, retry=retry)
                            if success:
                                success, message = self.set_context(completions_id=new_completions_id, mode="reset", new_messages=context, retry=retry)
        if success:
            message = f"Duplicated completions node '{completions_id}'."
        else:
            message = f"Failed to duplicate completions node '{completions_id}': {message}"

        return self._log_return(f"{new_completions_id}.duplicate", success, message, new_completions_id)

    def _release(self, completions_id, retry):
        assert_type_value(obj=completions_id, type_or_value=[None, str], name="argument 'completions_id'", logger=self._logger)

        request = CompletionsManage.Request()
        request.completions_id = "" if completions_id is None else completions_id
        request.action = "release"
        request.parameter_names = []
        request.parameter_values = []

        if completions_id is None:
            prefix = "completions.release"
        else:
            prefix = f"{completions_id}.release"

        success, message, _ = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_manage,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        return self._log_return(prefix, success, message)

    # Chat Completions API - Parameters

    def _get_parameters(self, completions_id, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)

        request = CompletionsSettingsGet.Request()
        request.completions_id = completions_id

        prefix = f"{completions_id}.get_parameters"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_get_settings,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        if success:
            parameters = {name: response.parameter_values[i] for i, name in enumerate(response.parameter_names)}
        else:
            parameters = None

        return self._log_return(prefix, success, message, parameters)

    def _reset_parameters(self, completions_id, retry):
        return self.set_parameters(completions_id=completions_id, parameter_names=[], parameter_values=[], retry=retry)

    def _set_parameters(self, completions_id, parameter_names, parameter_values, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=parameter_names, type_or_value=[str, list, None], name="argument 'parameter_names'", logger=self._logger)
        assert_type_value(obj=parameter_values, type_or_value=[str, list, None], name="argument 'parameter_values' (correct types are inferred from str)", logger=self._logger)

        if parameter_names is None:
            parameter_names = []
        elif isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        elif isinstance(parameter_names, list):
            for item in parameter_names:
                assert_type_value(obj=item, type_or_value=str, name="all elements in argument 'parameter_names'", logger=self._logger)

        if parameter_values is None:
            parameter_values = []
        elif isinstance(parameter_values, str):
            parameter_values = [parameter_values]
        elif isinstance(parameter_values, list):
            for item in parameter_values:
                assert_type_value(obj=item, type_or_value=str, name="all elements in argument 'parameter_values' (correct types are inferred)", logger=self._logger)

        assert_log(len(parameter_names) == len(parameter_values), f"Expected the number of provided 'parameter_names' ({len(parameter_names)}) and 'parameter_values' ({len(parameter_values)}) to match.", self._logger)

        request = CompletionsManage.Request()
        request.completions_id = completions_id
        request.action = "configure"
        request.parameter_names = parameter_names
        request.parameter_values = parameter_values

        prefix = f"{completions_id}.set_parameters"

        success, message, _ = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_manage,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        return self._log_return(prefix, success, message)

    def _async_set_parameters(self, completions_id, parameter_names, parameter_values, retry, succeed_async_id):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=parameter_names, type_or_value=[str, list, None], name="argument 'parameter_names'", logger=self._logger)
        assert_type_value(obj=parameter_values, type_or_value=[str, list, None], name="argument 'parameter_values' (correct types are inferred from str)", logger=self._logger)
        assert_type_value(obj=retry, type_or_value=[int, bool], name="argument 'retry'", logger=self._logger)
        assert_type_value(obj=succeed_async_id, type_or_value=[None, str], name="argument 'succeed_async_id'", logger=self._logger)

        if succeed_async_id is not None:
            assert_log(len(self._async_responses) > 0, f"Cannot register asynchronous thread succeeding ID '{succeed_async_id}' because no asynchronous threads have been started.", self._logger)
            assert_log(succeed_async_id in self._async_responses, f"Cannot register asynchronous thread succeeding unknown ID '{succeed_async_id}'. Known IDs: {list(self._async_responses.keys())}", self._logger)

        async_id = self._get_async_id(completions_id)

        self._async_responses[async_id] = {}
        self._async_responses[async_id]['type'] = "parameters"
        self._async_responses[async_id]['completions_id'] = completions_id
        self._async_responses[async_id]['succeed_async_id'] = succeed_async_id
        self._async_responses[async_id]['registered'] = self._node.get_clock().now()
        self._async_responses[async_id]['thread'] = threading.Thread(target=self._async_thread, args=(async_id, (completions_id, parameter_names, parameter_values, retry)))
        self._async_responses[async_id]['thread'].start()

        return self._log_return(f"{completions_id}.set_parameters", True, f"Registered asynchronous thread '{async_id}'.", async_id)

    # Chat Completions API - Prompting

    def _prompt(self, completions_id, text, role, reset_context, tool_response_id, response_type, identifier, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=text, type_or_value=[str, dict, list], name="argument 'text'", logger=self._logger)
        assert_type_value(obj=role, type_or_value=str, name="argument 'role'", logger=self._logger)
        assert_type_value(obj=reset_context, type_or_value=bool, name="argument 'reset_context'", logger=self._logger)
        assert_type_value(obj=tool_response_id, type_or_value=[None, str], name="argument 'tool_response_id'", logger=self._logger)
        assert_type_value(obj=response_type, type_or_value=[None, str], name="argument 'response_type'", logger=self._logger)
        assert_type_value(obj=identifier, type_or_value=[None, str], name="argument 'identifier'", logger=self._logger)

        if isinstance(text, (dict, list)):
            try:
                text = json.dumps(text)
            except Exception as e:
                assert_log(False, f"Failed to encode text '{text}' as JSON: {repr(e)}", self._logger)

        request = CompletionsPrompt.Request()
        request.completions_id = completions_id
        request.text = text
        request.role = role
        request.reset_context = reset_context
        request.tool_response_id = "" if tool_response_id is None else tool_response_id
        request.response_type = "none" if response_type is None else response_type
        request.identifier = "" if identifier is None else identifier

        prefix = f"{completions_id}.prompt"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_prompt,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            completion = json.loads(response.completion)
        else:
            completion = None

        return self._log_return(prefix, success, message, completion)

    def _async_prompt(self, completions_id, text, role, reset_context, tool_response_id, response_type, identifier, retry, succeed_async_id):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=text, type_or_value=[str, dict, list], name="argument 'text'", logger=self._logger)
        assert_type_value(obj=role, type_or_value=str, name="argument 'role'", logger=self._logger)
        assert_type_value(obj=reset_context, type_or_value=bool, name="argument 'reset_context'", logger=self._logger)
        assert_type_value(obj=tool_response_id, type_or_value=[None, str], name="argument 'tool_response_id'", logger=self._logger)
        assert_type_value(obj=response_type, type_or_value=[None, str], name="argument 'response_type'", logger=self._logger)
        assert_type_value(obj=identifier, type_or_value=[None, str], name="argument 'identifier'", logger=self._logger)
        assert_type_value(obj=retry, type_or_value=[int, bool], name="argument 'retry'", logger=self._logger)
        assert_type_value(obj=succeed_async_id, type_or_value=[None, str], name="argument 'succeed_async_id'", logger=self._logger)

        if succeed_async_id is not None:
            assert_log(len(self._async_responses) > 0, f"Cannot register asynchronous thread succeeding ID '{succeed_async_id}' because no asynchronous threads have been started.", self._logger)
            assert_log(succeed_async_id in self._async_responses, f"Cannot register asynchronous thread succeeding unknown ID '{succeed_async_id}'. Known IDs: {list(self._async_responses.keys())}", self._logger)

        async_id = self._get_async_id(completions_id)

        self._async_responses[async_id] = {}
        self._async_responses[async_id]['type'] = "prompt"
        self._async_responses[async_id]['completions_id'] = completions_id
        self._async_responses[async_id]['succeed_async_id'] = succeed_async_id
        self._async_responses[async_id]['registered'] = self._node.get_clock().now()
        self._async_responses[async_id]['thread'] = threading.Thread(target=self._async_thread, args=(async_id, (completions_id, text, role, reset_context, tool_response_id, response_type, identifier, retry)))
        self._async_responses[async_id]['thread'].start()

        return self._log_return(f"{completions_id}.prompt", True, f"Registered asynchronous thread '{async_id}'.", async_id)

    def _interrupt(self, completions_id, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)

        request = CompletionsInterrupt.Request()
        request.completions_id = completions_id

        prefix = f"{completions_id}.interrupt"

        success, message, _ = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_interrupt,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        return self._log_return(prefix, success, message)

    # Chat Completions API - Tools

    def _get_tools(self, completions_id, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)

        request = CompletionsToolsGet.Request()
        request.completions_id = completions_id

        prefix = f"{completions_id}.get_tools"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_get_tools,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        if success:
            tools = [json.loads(tool) for tool in response.tools]
        else:
            tools = None

        return self._log_return(prefix, success, message, tools)

    def _set_tools(self, completions_id, tools, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=tools, type_or_value=[None, list], name="argument 'tools'", logger=self._logger)

        tools_str = []
        if isinstance(tools, list):
            for tool in tools:
                assert_type_value(obj=tool, type_or_value=dict, name="argument 'succeed_async_id'", logger=self._logger)
                try:
                    tool_str = json.dumps(tool)
                except Exception as e:
                    assert_log(False, f"Failed to encode tool '{tool}' as JSON: {repr(e)}", self._logger)
                else:
                    tools_str.append(tool_str)

        request = CompletionsToolsSet.Request()
        request.completions_id = completions_id
        request.tools = tools_str

        prefix = f"{completions_id}.set_tools"

        success, message, _ = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_set_tools,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        return self._log_return(prefix, success, message)

    def _async_set_tools(self, completions_id, tools, retry, succeed_async_id):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=tools, type_or_value=[None, list], name="argument 'tools'", logger=self._logger)
        assert_type_value(obj=retry, type_or_value=[int, bool], name="argument 'retry'", logger=self._logger)
        assert_type_value(obj=succeed_async_id, type_or_value=[None, str], name="argument 'succeed_async_id'", logger=self._logger)

        if succeed_async_id is not None:
            assert_log(len(self._async_responses) > 0, f"Cannot register asynchronous thread succeeding ID '{succeed_async_id}' because no asynchronous threads have been started.", self._logger)
            assert_log(succeed_async_id in self._async_responses, f"Cannot register asynchronous thread succeeding unknown ID '{succeed_async_id}'. Known IDs: {list(self._async_responses.keys())}", self._logger)

        async_id = self._get_async_id(completions_id)

        self._async_responses[async_id] = {}
        self._async_responses[async_id]['type'] = "tools"
        self._async_responses[async_id]['completions_id'] = completions_id
        self._async_responses[async_id]['succeed_async_id'] = succeed_async_id
        self._async_responses[async_id]['registered'] = self._node.get_clock().now()
        self._async_responses[async_id]['thread'] = threading.Thread(target=self._async_thread, args=(async_id, (completions_id, tools, retry)))
        self._async_responses[async_id]['thread'].start()

        return self._log_return(f"{completions_id}.set_tools", True, f"Registered asynchronous thread '{async_id}'.", async_id)

    # Chat Completions API - Context

    def _get_context(self, completions_id, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)

        request = CompletionsContextGet.Request()
        request.completions_id = completions_id

        prefix = f"{completions_id}.get_context"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_get_context,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        if success:
            context = [json.loads(msg) for msg in response.context]
        else:
            context = None

        return self._log_return(prefix, success, message, context)

    def _set_context(self, completions_id, mode, new_messages, index, indexing_last_to_first, retry):
        assert_type_value(obj=completions_id, type_or_value=str, name="argument 'completions_id'", logger=self._logger)
        assert_type_value(obj=mode, type_or_value=str, name="argument 'mode'", logger=self._logger)
        assert_type_value(obj=new_messages, type_or_value=[None, list], name="argument 'new_messages'", logger=self._logger)
        assert_type_value(obj=index, type_or_value=int, name="argument 'index'", logger=self._logger)
        assert_type_value(obj=indexing_last_to_first, type_or_value=bool, name="argument 'indexing_last_to_first'", logger=self._logger)

        new_messages_str = []
        if isinstance(new_messages, list):
            for message in new_messages:
                assert_type_value(obj=message, type_or_value=dict, name="argument 'message'", logger=self._logger)
                try:
                    message_str = json.dumps(message)
                except Exception as e:
                    raise Exception(f"Provided argument 'new_messages' contains element that cannot be parsed as JSON: {repr(e)}")
                else:
                    new_messages_str.append(message_str)

        request = CompletionsContextSet.Request()
        request.completions_id = completions_id
        request.mode = mode
        request.new_messages = new_messages_str
        request.index = abs(index)
        request.indexing_last_to_first = indexing_last_to_first

        prefix = f"{completions_id}.set_context"

        success, message, _ = self._client_wrapper(
            prefix=prefix,
            client=self._cli_completions_set_context,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        return self._log_return(prefix, success, message)

    # Embeddings API

    def _get_embeddings(self, text, identifier, retry):
        assert_type_value(obj=text, type_or_value=[str, list], name="argument 'text'", logger=self._logger)
        assert_type_value(obj=identifier, type_or_value=[None, str], name="argument 'identifier'", logger=self._logger)

        if isinstance(text, str):
            text_list = [text]
        else:
            for t in text:
                assert_type_value(obj=t, type_or_value=str, name="element in argument 'text'", logger=self._logger)
            text_list = text

        request = EmbeddingsGet.Request()
        request.texts = text_list
        request.identifier = "" if identifier is None else identifier

        prefix = "embeddings"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_get_embeddings,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            embeddings = [list(embedding.embedding) for embedding in response.embeddings]
            if isinstance(text, str):
                embeddings = embeddings[0]
        else:
            embeddings = None

        return self._log_return(prefix, success, message, embeddings)

    # Images API

    def _get_images(self, prompt, model, quality, style, size, retry):
        assert_type_value(obj=prompt, type_or_value=str, name="argument 'prompt'", logger=self._logger)
        assert_type_value(obj=model, type_or_value=[None, str], name="argument 'model'", logger=self._logger)
        assert_type_value(obj=quality, type_or_value=[None, str], name="argument 'quality'", logger=self._logger)
        assert_type_value(obj=style, type_or_value=[None, str], name="argument 'style'", logger=self._logger)
        assert_type_value(obj=size, type_or_value=[None, str], name="argument 'size'", logger=self._logger)

        request = ImagesGet.Request()
        request.prompt = prompt
        request.model = "" if model is None else model
        request.quality = "" if quality is None else quality
        request.style = "" if style is None else style
        request.size = "" if size is None else size

        prefix = "images"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_get_images,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            path = response.path
        else:
            path = None

        return self._log_return(prefix, success, message, path)

    # Speech API

    def _get_speech(self, text, model, voice, speed, instructions, retry):
        assert_type_value(obj=text, type_or_value=str, name="argument 'text'", logger=self._logger)
        assert_type_value(obj=model, type_or_value=[None, str], name="argument 'model'", logger=self._logger)
        assert_type_value(obj=voice, type_or_value=[None, str], name="argument 'voice'", logger=self._logger)
        assert_type_value(obj=speed, type_or_value=[float, int], name="argument 'speed'", logger=self._logger)
        assert_type_value(obj=instructions, type_or_value=[None, str], name="argument 'instructions'", logger=self._logger)

        prefix = "speech"

        if instructions in self._settings['voice_presets']:
            if voice is None or voice == "":
                self._logger.debug(f"{prefix} Using voice and instructions from preset '{instructions}'")
                instructions = self._settings['voice_presets'][instructions]['instructions']
                voice = self._settings['voice_presets'][instructions]['voice']
            else:
                self._logger.debug(f"{prefix} Using instructions from preset '{instructions}'")
                instructions = self._settings['voice_presets'][instructions]['instructions']

        request = request = SpeechGet.Request()
        request.text = text
        request.model = "" if model is None else model
        request.voice = "" if voice is None else voice
        request.speed = float(speed)
        request.instructions = "" if instructions is None else instructions

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_get_speech,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            path = response.path
        else:
            path = None

        return self._log_return(prefix, success, message, path)

    # NimbRo Vision API

    def _mmgroundingdino(self, image, prompts, model_id, model_flavor, min_confidence, nms_iou, overdetect_factor, retry):
        assert_type_value(obj=image, type_or_value=[list, str], name="argument 'image'", logger=self._logger)
        assert_type_value(obj=prompts, type_or_value=list, name="argument 'prompt'", logger=self._logger)
        assert_type_value(obj=model_id, type_or_value=int, name="argument 'model_id'", logger=self._logger)
        assert_type_value(obj=model_flavor, type_or_value=str, name="argument 'model_flavor'", logger=self._logger)
        assert_type_value(obj=min_confidence, type_or_value=[list, float], name="argument 'min_confidence'", logger=self._logger)
        assert_type_value(obj=nms_iou, type_or_value=[list, float, None], name="argument 'nms_iou'", logger=self._logger)
        assert_type_value(obj=overdetect_factor, type_or_value=[list, float, None], name="argument 'overdetect_factor'", logger=self._logger)

        batch = False

        if isinstance(image, list):
            batch = True
            for item in image:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'image'", logger=self._logger)

        all_lists, all_str = True, True
        for prompt in prompts:
            if isinstance(prompt, list):
                all_str = False
            elif isinstance(prompt, str):
                all_lists = False
            else:
                assert_log(False, f"Provided argument 'prompts' contains element of invalid type '{type(prompt).__name__}'. Supported types are 'list' and 'str'.", self._logger)
        assert_log(all_lists or all_str, "Provided argument 'prompts' contains element of mixed types 'list' and 'str' but all elements must be either 'list' or 'str'.", self._logger)
        if all_lists:
            batch = True
            assert_log(
                all(all(isinstance(prompt, str) for prompt in prompts_image) for prompts_image in prompts),
                f"Provided argument 'prompts' is list of lists that contains invalid type {[[type(prompt).__name__ for prompt in image_prompts] for image_prompts in prompts]}. Supported type is 'str'.",
                self._logger
            )
            assert_log(len(prompts) > 0, f"Provided argument 'prompts' is list of lists {[len(prompts_image) for prompts_image in prompts]} where one of them is empty.", self._logger)
            assert_log(
                not any(len(prompts_image) == 0 for prompts_image in prompts),
                f"Provided argument 'prompts' is list of lists {[len(prompts_image) for prompts_image in prompts]} where one of them is empty.",
                self._logger
            )
            if len(prompts) == 1:
                all_lists = False
                prompts = prompts[0]

        if isinstance(min_confidence, list):
            batch = True
            for item in min_confidence:
                assert_type_value(obj=item, type_or_value=float, name="element in argument 'min_confidence'", logger=self._logger)
            if len(min_confidence) == 1:
                min_confidence = min_confidence[0]

        if isinstance(nms_iou, list):
            batch = True
            for item in nms_iou:
                assert_type_value(obj=item, type_or_value=[None, float], name="element in argument 'nms_iou'", logger=self._logger)
            if len(nms_iou) == 1:
                nms_iou = nms_iou[0]

        if isinstance(overdetect_factor, list):
            batch = True
            for item in overdetect_factor:
                assert_type_value(obj=item, type_or_value=[None, float], name="element in argument 'overdetect_factor'", logger=self._logger)
            if len(overdetect_factor) == 1:
                overdetect_factor = overdetect_factor[0]

        num_settings = None
        if all_lists:
            num_settings = len(prompts)
        if isinstance(min_confidence, list) and len(min_confidence) > 1:
            if num_settings is None:
                num_settings = len(min_confidence)
            else:
                assert_log(num_settings == len(min_confidence), f"Provided argument 'min_confidence' is list of length '{len(min_confidence)}' which cannot be broadcasted with another parameter provided as list of length '{num_settings}'.", self._logger)
        if isinstance(nms_iou, list) and len(nms_iou) > 1:
            if num_settings is None:
                num_settings = len(nms_iou)
            else:
                assert_log(num_settings == len(nms_iou), f"Provided argument 'nms_iou' is list of length '{len(nms_iou)}' which cannot be broadcasted with another parameter provided as list of length '{num_settings}'.", self._logger)
        if isinstance(overdetect_factor, list) and len(overdetect_factor) > 1:
            if num_settings is None:
                num_settings = len(overdetect_factor)
            else:
                assert_log(num_settings == len(overdetect_factor), f"Provided argument 'overdetect_factor' is list of length '{len(overdetect_factor)}' which cannot be broadcasted with another parameter provided as list of length '{num_settings}'.", self._logger)
        if isinstance(image, list) and len(image) > 1 and num_settings is not None and num_settings > 1:
            assert_log(len(image) == num_settings, f"Provided argument 'image' contains '{len(image)}' images, which cannot be broadcasted to the number of settings '{num_settings}'.", self._logger)

        data = {}
        if isinstance(image, list):
            data['images'] = image
        else:
            data['images'] = [image]
        if num_settings is None:
            data['inference_parameters'] = [{'prompts': prompts, 'min_confidence': min_confidence, 'nms_iou': nms_iou, 'overdetect_factor': overdetect_factor}]
        else:
            data['inference_parameters'] = []
            for i in range(num_settings):
                if all_lists:
                    prompts_set = prompts[i]
                else:
                    prompts_set = prompts
                if isinstance(min_confidence, list):
                    min_confidence_set = min_confidence[i]
                else:
                    min_confidence_set = min_confidence
                if isinstance(overdetect_factor, list):
                    overdetect_factor_set = overdetect_factor[i]
                else:
                    overdetect_factor_set = overdetect_factor
                data['inference_parameters'].append({'prompts': prompts_set, 'min_confidence': min_confidence_set, 'overdetect_factor': overdetect_factor_set})
        data = json.dumps(data)

        request = NimbroVisionGet.Request()
        request.model_id = abs(model_id)
        request.flavor = model_flavor
        request.data = data

        prefix = "vision.mmgd"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_mmgroundingdino,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            result = json.loads(response.result)
            result = result['artifact']['detections']
            if not batch:
                result = result[0]
        else:
            result = None

        return self._log_return(prefix, success, message, result)

    def _sam2_realtime_update(self, image, prompts, model_id, model_flavor, retry):
        assert_type_value(obj=image, type_or_value=str, name="argument 'image'", logger=self._logger)
        assert_type_value(obj=prompts, type_or_value=list, name="argument 'prompts'", logger=self._logger)
        assert_type_value(obj=model_id, type_or_value=int, name="argument 'model_id'", logger=self._logger)
        assert_type_value(obj=model_flavor, type_or_value=str, name="argument 'model_flavor'", logger=self._logger)

        for prompt in prompts:
            assert_type_value(obj=prompt, type_or_value=dict, name="element in argument 'prompts'", logger=self._logger)
        try:
            json.dumps(prompts)
        except Exception as e:
            assert_log(False, f"Provided argument 'prompts' cannot be parsed as JSON: {repr(e)}", self._logger)

        data = json.dumps({'image': image, 'prompts': prompts})

        request = NimbroVisionGet.Request()
        request.model_id = abs(model_id)
        request.flavor = model_flavor
        request.data = data

        prefix = "vision.sam2_update"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_sam2_realtime_update,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            result = json.loads(response.result)
            result = result['artifact']['tracks']
            result = result[0]
        else:
            result = None

        return self._log_return(prefix, success, message, result)

    def _sam2_realtime_track(self, image, model_id, retry):
        assert_type_value(obj=image, type_or_value=[list, str], name="argument 'image'", logger=self._logger)
        assert_type_value(obj=model_id, type_or_value=int, name="argument 'model_id'", logger=self._logger)

        batch = False

        if isinstance(image, list):
            batch = True
            for item in image:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'image'", logger=self._logger)

        data = json.dumps({'images': image} if isinstance(image, list) else {'images': [image]})

        request = NimbroVisionGet.Request()
        request.model_id = abs(model_id)
        request.flavor = ""
        request.data = data

        prefix = "vision.sam2_track"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_sam2_realtime_track,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            result = json.loads(response.result)
            result = result['artifact']['tracks']
            if not batch:
                result = result[0]
        else:
            result = None

        return self._log_return(prefix, success, message, result)

    def _dam(self, image, prompts, query, model_id, model_flavor, temperature, top_p, num_beams, max_new_tokens, max_batch_size, retry):
        assert_type_value(obj=image, type_or_value=[list, str], name="argument 'image'", logger=self._logger)
        assert_type_value(obj=prompts, type_or_value=[list, dict], name="argument 'prompt'", logger=self._logger)
        assert_type_value(obj=query, type_or_value=[list, str], name="argument 'query'", logger=self._logger)
        assert_type_value(obj=model_id, type_or_value=int, name="argument 'model_id'", logger=self._logger)
        assert_type_value(obj=model_flavor, type_or_value=str, name="argument 'model_flavor'", logger=self._logger)
        assert_type_value(obj=temperature, type_or_value=[list, float], name="argument 'temperature'", logger=self._logger)
        assert_type_value(obj=top_p, type_or_value=[list, float], name="argument 'top_p'", logger=self._logger)
        assert_type_value(obj=num_beams, type_or_value=[list, int], name="argument 'num_beams'", logger=self._logger)
        assert_type_value(obj=max_new_tokens, type_or_value=[list, int], name="argument 'max_new_tokens'", logger=self._logger)
        assert_type_value(obj=max_batch_size, type_or_value=[list, int], name="argument 'max_batch_size'", logger=self._logger)

        batch = False

        if isinstance(image, list):
            batch = True
            for item in image:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'image'", logger=self._logger)

        all_lists, all_dict = True, True
        for prompt in prompts:
            if isinstance(prompt, list):
                all_dict = False
            elif isinstance(prompt, dict):
                all_lists = False
            else:
                assert_log(False, f"Provided argument 'prompts' contains element of invalid type '{type(prompt).__name__}'. Supported types are 'list' and 'dict'.", self._logger)
        assert_log(all_lists or all_dict, "Provided argument 'prompts' contains element of mixed types 'list' and 'dict' but all elements must be either 'list' or 'dict'.", self._logger)
        if all_lists:
            batch = True
            assert_log(
                all(all(isinstance(prompt, dict) for prompt in prompts_image) for prompts_image in prompts),
                f"Provided argument 'prompts' is list of lists that contains invalid type {[[type(prompt).__name__ for prompt in image_prompts] for image_prompts in prompts]}. Supported type is 'dict'.",
                self._logger
            )
            assert_log(len(prompts) > 0, f"Provided argument 'prompts' is list of lists {[len(prompts_image) for prompts_image in prompts]} where one of them is empty.", self._logger)
            assert_log(
                not any(len(prompts_image) == 0 for prompts_image in prompts),
                f"Provided argument 'prompts' is list of lists {[len(prompts_image) for prompts_image in prompts]} where one of them is empty.",
                self._logger
            )
            if len(prompts) == 1:
                all_lists = False
                prompts = prompts[0]
        try:
            json.dumps(prompts)
        except Exception as e:
            assert_log(False, f"Provided argument 'prompts' cannot be parsed as JSON: {repr(e)}", self._logger)

        if isinstance(query, list):
            batch = True
            for item in query:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'query'", logger=self._logger)
            if len(query) == 1:
                query = query[0]

        if isinstance(temperature, list):
            batch = True
            for item in temperature:
                assert_type_value(obj=item, type_or_value=float, name="element in argument 'temperature'", logger=self._logger)
            if len(temperature) == 1:
                temperature = temperature[0]

        if isinstance(top_p, list):
            batch = True
            for item in top_p:
                assert_type_value(obj=item, type_or_value=float, name="element in argument 'top_p'", logger=self._logger)
            if len(top_p) == 1:
                top_p = top_p[0]

        if isinstance(num_beams, list):
            batch = True
            for item in num_beams:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'num_beams'", logger=self._logger)
            if len(num_beams) == 1:
                num_beams = num_beams[0]

        if isinstance(max_new_tokens, list):
            batch = True
            for item in max_new_tokens:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'max_new_tokens'", logger=self._logger)
            if len(max_new_tokens) == 1:
                max_new_tokens = max_new_tokens[0]

        if isinstance(max_batch_size, list):
            batch = True
            for item in max_batch_size:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'max_batch_size'", logger=self._logger)
            if len(max_batch_size) == 1:
                max_batch_size = max_batch_size[0]

        lengths = [
            len(temperature) if isinstance(temperature, list) else 1,
            len(top_p) if isinstance(top_p, list) else 1,
            len(num_beams) if isinstance(num_beams, list) else 1,
            len(max_new_tokens) if isinstance(max_new_tokens, list) else 1,
            len(max_batch_size) if isinstance(max_batch_size, list) else 1
        ]
        max_length = max(lengths)
        assert_log(max_length > 0, "Expected all arguments 'temperature', 'top_p', 'num_beams', 'max_new_tokens', and 'max_batch_size' to specify at least one value.", self._logger)
        assert_log(
            all(x == max_length or x == 1 for x in lengths),
            f"Expected all arguments of the listed arguments to specify one or '{max_length}' values but got: {dict(zip(['temperature', 'top_p', 'num_beams', 'max_new_tokens', 'max_batch_size'], lengths))}",
            self._logger
        )

        inference_parameters = []
        for i in range(max_length):
            parameter_set = {}
            for j, (parameter, value) in enumerate(zip(["temperature", "top_p", "num_beams", "max_new_tokens", "max_batch_size"], [temperature, top_p, num_beams, max_new_tokens, max_batch_size])):
                if lengths[j] == 1:
                    parameter_set[parameter] = value
                else:
                    parameter_set[parameter] = value[i]
            inference_parameters.append(parameter_set)

        data = json.dumps({'images': image if isinstance(image, list) else [image], 'prompts': prompts if all_lists else [prompts], 'queries': query if isinstance(query, list) else [query], 'inference_parameters': inference_parameters})

        request = NimbroVisionGet.Request()
        request.model_id = abs(model_id)
        request.flavor = model_flavor
        request.data = data

        prefix = "vision.dam"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_dam,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            result = json.loads(response.result)
            result = result['artifact']['descriptions']
            if not batch:
                result = result[0]
        else:
            result = None

        return self._log_return(prefix, success, message, result)

    def _kosmos2(self, image, prompt, model_id, model_flavor, num_beams, max_new_tokens, max_batch_size, retry):
        assert_type_value(obj=image, type_or_value=[list, str], name="argument 'image'", logger=self._logger)
        assert_type_value(obj=prompt, type_or_value=[list, str], name="argument 'prompt'", logger=self._logger)
        assert_type_value(obj=model_id, type_or_value=int, name="argument 'model_id'", logger=self._logger)
        assert_type_value(obj=model_flavor, type_or_value=str, name="argument 'model_flavor'", logger=self._logger)
        assert_type_value(obj=num_beams, type_or_value=[list, int], name="argument 'num_beams'", logger=self._logger)
        assert_type_value(obj=max_new_tokens, type_or_value=[list, int], name="argument 'max_new_tokens'", logger=self._logger)
        assert_type_value(obj=max_batch_size, type_or_value=[list, int], name="argument 'max_batch_size'", logger=self._logger)

        batch = False

        if isinstance(image, list):
            batch = True
            for item in image:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'image'", logger=self._logger)

        if isinstance(prompt, list):
            batch = True
            for item in prompt:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'prompt'", logger=self._logger)
        else:
            prompt = [prompt]

        if isinstance(num_beams, list):
            batch = True
            for item in num_beams:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'num_beams'", logger=self._logger)
            if len(num_beams) == 1:
                num_beams = num_beams[0]

        if isinstance(max_new_tokens, list):
            batch = True
            for item in max_new_tokens:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'max_new_tokens'", logger=self._logger)
            if len(max_new_tokens) == 1:
                max_new_tokens = max_new_tokens[0]

        if isinstance(max_batch_size, list):
            batch = True
            for item in max_batch_size:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'max_batch_size'", logger=self._logger)
            if len(max_batch_size) == 1:
                max_batch_size = max_batch_size[0]

        lengths = [
            len(num_beams) if isinstance(num_beams, list) else 1,
            len(max_new_tokens) if isinstance(max_new_tokens, list) else 1,
            len(max_batch_size) if isinstance(max_batch_size, list) else 1
        ]
        max_length = max(lengths)
        assert_log(max_length > 0, "Expected all arguments 'num_beams', 'max_new_tokens', and 'max_batch_size' to specify at least one value.", self._logger)
        assert_log(
            all(x == max_length or x == 1 for x in lengths),
            f"Expected all arguments of the listed arguments to specify one or '{max_length}' values but got: {dict(zip(['num_beams', 'max_new_tokens', 'max_batch_size'], lengths))}",
            self._logger
        )

        inference_parameters = []
        for i in range(max_length):
            parameter_set = {}
            for j, (parameter, value) in enumerate(zip(["num_beams", "max_new_tokens", "max_batch_size"], [num_beams, max_new_tokens, max_batch_size])):
                if lengths[j] == 1:
                    parameter_set[parameter] = value
                else:
                    parameter_set[parameter] = value[i]
            inference_parameters.append(parameter_set)

        data = json.dumps({'images': image if isinstance(image, list) else [image], 'prompts': prompt, 'inference_parameters': inference_parameters[0]})

        request = NimbroVisionGet.Request()
        request.model_id = abs(model_id)
        request.flavor = model_flavor
        request.data = data

        prefix = "vision.kosmos2"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_kosmos2,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            result = json.loads(response.result)
            detections = result['artifact']['detections']
            captions = result['artifact']['captions']
            if not batch:
                detections = detections[0]
                captions = captions[0]
        else:
            detections, captions = None, None

        return self._log_return(prefix, success, message, detections, captions)

    def _florence2(self, image, prompt, model_id, model_flavor, num_beams, max_new_tokens, max_batch_size, retry):
        assert_type_value(obj=image, type_or_value=[list, str], name="argument 'image'", logger=self._logger)
        assert_type_value(obj=prompt, type_or_value=[list, dict], name="argument 'prompt'", logger=self._logger)
        assert_type_value(obj=model_id, type_or_value=int, name="argument 'model_id'", logger=self._logger)
        assert_type_value(obj=model_flavor, type_or_value=str, name="argument 'model_flavor'", logger=self._logger)
        assert_type_value(obj=num_beams, type_or_value=[list, int], name="argument 'num_beams'", logger=self._logger)
        assert_type_value(obj=max_new_tokens, type_or_value=[list, int], name="argument 'max_new_tokens'", logger=self._logger)
        assert_type_value(obj=max_batch_size, type_or_value=[list, int], name="argument 'max_batch_size'", logger=self._logger)

        batch = False

        if isinstance(image, list):
            batch = True
            for item in image:
                assert_type_value(obj=item, type_or_value=str, name="element in argument 'image'", logger=self._logger)

        if isinstance(prompt, list):
            batch = True
            for item in prompt:
                assert_type_value(obj=item, type_or_value=dict, name="element in argument 'prompt'", logger=self._logger)
        else:
            prompt = [copy.deepcopy(prompt)]

        if isinstance(num_beams, list):
            batch = True
            for item in num_beams:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'num_beams'", logger=self._logger)
            if len(num_beams) == 1:
                num_beams = num_beams[0]

        if isinstance(max_new_tokens, list):
            batch = True
            for item in max_new_tokens:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'max_new_tokens'", logger=self._logger)
            if len(max_new_tokens) == 1:
                max_new_tokens = max_new_tokens[0]

        if isinstance(max_batch_size, list):
            batch = True
            for item in max_batch_size:
                assert_type_value(obj=item, type_or_value=int, name="element in argument 'max_batch_size'", logger=self._logger)
            if len(max_batch_size) == 1:
                max_batch_size = max_batch_size[0]

        lengths = [
            len(num_beams) if isinstance(num_beams, list) else 1,
            len(max_new_tokens) if isinstance(max_new_tokens, list) else 1,
            len(max_batch_size) if isinstance(max_batch_size, list) else 1
        ]
        max_length = max(lengths)
        assert_log(max_length > 0, "Expected all arguments 'num_beams', 'max_new_tokens', and 'max_batch_size' to specify at least one value.", self._logger)
        assert_log(
            all(x == max_length or x == 1 for x in lengths),
            f"Expected all arguments of the listed arguments to specify one or '{max_length}' values but got: {dict(zip(['num_beams', 'max_new_tokens', 'max_batch_size'], lengths))}",
            self._logger
        )

        inference_parameters = []
        for i in range(max_length):
            parameter_set = {}
            for j, (parameter, value) in enumerate(zip(["num_beams", "max_new_tokens", "max_batch_size"], [num_beams, max_new_tokens, max_batch_size])):
                if lengths[j] == 1:
                    parameter_set[parameter] = value
                else:
                    parameter_set[parameter] = value[i]
            inference_parameters.append(parameter_set)

        data = json.dumps({'images': image if isinstance(image, list) else [image], 'prompts': prompt, 'inference_parameters': inference_parameters[0]})

        request = NimbroVisionGet.Request()
        request.model_id = abs(model_id)
        request.flavor = model_flavor
        request.data = data

        prefix = "vision.florence2"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_florence2,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_response'],
            retry=retry
        )

        if success:
            result = json.loads(response.result)
            detections = result['artifact']['detections']
            captions = result['artifact']['captions']
            if not batch:
                detections = detections[0]
                captions = captions[0]
        else:
            detections, captions = None, None

        return self._log_return(prefix, success, message, detections, captions)

    # General

    def _get_usage(self, api_type, api_endpoint, model_name, identifier, stamp_start, stamp_end, retry):
        assert_type_value(obj=api_type, type_or_value=[None, str], name="argument 'api_type'", logger=self._logger)
        assert_type_value(obj=api_endpoint, type_or_value=[None, str], name="argument 'api_endpoint'", logger=self._logger)
        assert_type_value(obj=model_name, type_or_value=[None, str], name="argument 'model_name'", logger=self._logger)
        assert_type_value(obj=identifier, type_or_value=[None, str], name="argument 'identifier'", logger=self._logger)
        assert_type_value(stamp_start, [None, float, int, str, datetime.datetime, rclpy.time.Time, builtin_interfaces.msg.Time], name="argument 'stamp_start'", logger=self._logger)
        assert_type_value(stamp_end, [None, float, int, str, datetime.datetime, rclpy.time.Time, builtin_interfaces.msg.Time], name="argument 'stamp_end'", logger=self._logger)

        if stamp_start is None:
            stamp_start = ""
        elif not isinstance(stamp_start, str):
            stamp_start = convert_stamp(stamp=stamp_start, target_format="iso")

        if stamp_end is None:
            stamp_end = ""
        elif not isinstance(stamp_end, str):
            stamp_end = convert_stamp(stamp=stamp_end, target_format="iso")

        request = UsageGet.Request()
        request.api_type = "" if api_type is None else api_type
        request.api_endpoint = "" if api_endpoint is None else api_endpoint
        request.model_name = "" if model_name is None else model_name
        request.identifier = "" if identifier is None else identifier
        request.stamp_start = stamp_start
        request.stamp_end = stamp_end

        prefix = "usage"

        success, message, response = self._client_wrapper(
            prefix=prefix,
            client=self._cli_get_usage,
            request=request,
            timeout_service=self._settings['timeout_service'],
            timeout_response=self._settings['timeout_service'],
            retry=retry
        )

        if success:
            usage = json.loads(response.usage)
        else:
            usage = None

        return self._log_return(prefix, success, message, usage)

    def _async_get(self, async_id, mute_timeout_logging, timeout):
        assert_type_value(obj=async_id, type_or_value=str, name="argument 'async_id'", logger=self._logger)
        assert_log(len(self._async_responses) > 0, "Cannot retrieve asynchronous response because no asynchronous threads have been started.", self._logger)
        assert_log(async_id in self._async_responses, f"Cannot retrieve asynchronous response for unknown ID '{async_id}'. Known IDs: {list(self._async_responses.keys())}", self._logger)
        assert_type_value(obj=mute_timeout_logging, type_or_value=bool, name="argument 'mute_timeout_logging'", logger=self._logger)
        assert_type_value(obj=timeout, type_or_value=[None, float, int], name="argument 'timeout'", logger=self._logger)

        if timeout is not None:
            timeout = abs(timeout)

        prefix = f"completions.async_get.{async_id}"

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
                    self._logger.info(f"[{prefix}] Waiting for response from asynchronous thread since '{time_waited:.3f}s'.", throttle_duration_sec=1.0)
                else:
                    message = f"Retrieved response from asynchronous thread after waiting '{time_waited:.3f}s'."
                    break
        else:
            message = "Retrieved response from asynchronous thread without waiting."

        if 'received' not in self._async_responses[async_id]:
            self._async_responses[async_id]['received'] = now

        return self._log_return(prefix, True, message, self._async_responses[async_id]['response'])

    def _get_async_status(self):
        if len(self._async_responses) == 0:
            self._logger.info("No asynchronous threads have been registered.")
        else:
            self._logger.info("Asynchronous thread info:")
            for result in self._async_responses:
                self._logger.info(f"ID '{result}': {self._async_responses[result]}")
