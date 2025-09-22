#!/usr/bin/env python3

import json
import threading

import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.srv import SetParametersAtomically, GetParameters
from rcl_interfaces.msg import ParameterType

from nimbro_api_interfaces.srv import CompletionsManage, CompletionsStatusGet, CompletionsSettingsGet
from nimbro_api_interfaces.srv import CompletionsPrompt, CompletionsInterrupt, CompletionsToolsGet, CompletionsToolsSet, CompletionsContextGet, CompletionsContextSet, TriggerFeedback

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, SelfShutdown, block_until_future_complete

### <Parameter Defaults>

node_name = "completions_multiplexer"
severity = 10

managed_nodes = [""]

timeout_service = 5.0 # seconds
timeout_completion = 500.0 # seconds

## non-params

acquire_style = 1 # When forwarding to completions node that is not acquired: Just warn (0), Acquire (1), Block (2)
status_interval = 1.0 # seconds

### </Parameter Defaults>

class CompletionsMultiplexer(Node):

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
            name="managed_nodes",
            dtype=list[str],
            default_value=managed_nodes,
            description="Names of the completions nodes to be managed.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="timeout_service",
            dtype=float,
            default_value=timeout_service,
            description="Time in seconds waited for basic responses from service request.",
            read_only=False,
            range_min=0.1,
            range_max=86400.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="timeout_completion",
            dtype=float,
            default_value=timeout_completion,
            description="Time in seconds waited until a Chat Completion is finished.",
            read_only=False,
            range_min=0.1,
            range_max=86400.0,
            range_step=0.0
        )

        self.completions = {}
        for n in self.parameters.managed_nodes:
            if n == "":

                continue
            self.completions[n] = {}
            self.completions[n]['locked'] = False

            cbg_prompt = MutuallyExclusiveCallbackGroup()

            self.completions[n]['prompt'] = self.create_client(CompletionsPrompt, f"/{n}/prompt".replace("//", "/"), callback_group=cbg_prompt)
            self.completions[n]['interrupt'] = self.create_client(CompletionsInterrupt, f"/{n}/interrupt".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())

            self.completions[n]['get_tools'] = self.create_client(CompletionsToolsGet, f"/{n}/get_tools".replace("//", "/"), callback_group=cbg_prompt)
            self.completions[n]['set_tools'] = self.create_client(CompletionsToolsSet, f"/{n}/set_tools".replace("//", "/"), callback_group=cbg_prompt)

            self.completions[n]['get_context'] = self.create_client(CompletionsContextGet, f"/{n}/get_context".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())
            self.completions[n]['set_context'] = self.create_client(CompletionsContextSet, f"/{n}/set_context".replace("//", "/"), callback_group=cbg_prompt)

            self.completions[n]['get_parameters'] = self.create_client(GetParameters, f"/{n}/get_parameters".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())
            self.completions[n]['set_parameters'] = self.create_client(SetParametersAtomically, f"/{n}/set_parameters_atomically".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())
            self.completions[n]['reset'] = self.create_client(TriggerFeedback, f"/{n}/reset_parameters".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())

        self.valid_completions_parameters = {
            'severity': ParameterType.PARAMETER_INTEGER,
            'log_line_length': ParameterType.PARAMETER_INTEGER,
            'log_last_messages': ParameterType.PARAMETER_INTEGER,
            'log_chunks': ParameterType.PARAMETER_BOOL,
            'probe_api_connection': ParameterType.PARAMETER_BOOL,
            'api_endpoint': ParameterType.PARAMETER_STRING,
            'model_name': ParameterType.PARAMETER_STRING,
            'model_temperature': ParameterType.PARAMETER_DOUBLE,
            'model_top_p': ParameterType.PARAMETER_DOUBLE,
            'model_max_tokens': ParameterType.PARAMETER_INTEGER,
            'model_presence_penalty': ParameterType.PARAMETER_DOUBLE,
            'model_frequency_penalty': ParameterType.PARAMETER_DOUBLE,
            'model_reasoning_effort': ParameterType.PARAMETER_STRING,
            'completion_parsers': ParameterType.PARAMETER_STRING_ARRAY,
            'completion_parsers_timeout': ParameterType.PARAMETER_DOUBLE,
            'completion_parsers_folder': ParameterType.PARAMETER_STRING,
            'stream_completion': ParameterType.PARAMETER_BOOL,
            'normalize_text_completion': ParameterType.PARAMETER_BOOL,
            'max_tool_calls_per_completion': ParameterType.PARAMETER_INTEGER,
            'correction_attempts': ParameterType.PARAMETER_INTEGER,
            'timeout_chunk_first': ParameterType.PARAMETER_DOUBLE,
            'timeout_chunk_next': ParameterType.PARAMETER_DOUBLE,
            'timeout_completion': ParameterType.PARAMETER_DOUBLE
        }
        self.completions_parameters_exclude_from_get = ["severity", "log_line_length", "log_last_messages", "log_chunks"]

        self.lock = threading.Lock()

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=50)

        self.srv_manage = self.create_service(CompletionsManage, f"{self.node_namespace}/{self.node_name}/manage".replace("//", "/"), self.manage_completions, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_status = self.create_service(CompletionsStatusGet, f"{self.node_namespace}/{self.node_name}/get_status".replace("//", "/"), self.get_status, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_settings = self.create_service(CompletionsSettingsGet, f"{self.node_namespace}/{self.node_name}/get_settings".replace("//", "/"), self.get_completions_settings, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())

        self.srv_prompt = self.create_service(CompletionsPrompt, f"{self.node_namespace}/{self.node_name}/prompt".replace("//", "/"), self.forward_prompt, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_interrupt = self.create_service(CompletionsInterrupt, f"{self.node_namespace}/{self.node_name}/interrupt".replace("//", "/"), self.forward_interrupt, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_tools = self.create_service(CompletionsToolsGet, f"{self.node_namespace}/{self.node_name}/get_tools".replace("//", "/"), self.forward_get_tools, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_set_tools = self.create_service(CompletionsToolsSet, f"{self.node_namespace}/{self.node_name}/set_tools".replace("//", "/"), self.forward_set_tools, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_context = self.create_service(CompletionsContextGet, f"{self.node_namespace}/{self.node_name}/get_context".replace("//", "/"), self.forward_get_context, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_set_context = self.create_service(CompletionsContextSet, f"{self.node_namespace}/{self.node_name}/set_context".replace("//", "/"), self.forward_set_context, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())

        self.pub_status = self.create_publisher(DiagnosticStatus, f"{self.node_namespace}/{self.node_name}/status".replace("//", "/"), qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.timer_status = self.create_timer(status_interval, self.publish_status, callback_group=MutuallyExclusiveCallbackGroup())

        self._logger.info("Node started")

    def __del__(self):
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self._logger.set_settings(settings={'severity': value})

        elif name == "managed_nodes":
            if len(value) == 0:
                value = None
                message = "At least one completions node must be specified."

        return value, message

    # Completions Allocation

    def manage_completions(self, request, response):
        response.success = True
        response.message = ""
        response.completions_id = ""

        if request.action == "acquire":

            if request.completions_id != "":
                self._logger.warn(f"Non-empty field completions_id '{request.completions_id}' is being ignored while acquiring completions node.")
                request.completions_id = ""

            self.lock.acquire()
            for n in self.parameters.managed_nodes:
                if n == "":
                    continue
                if not self.completions[n]['locked']:
                    self.completions[n]['locked'] = True
                    response.message = f"Acquired completions node '{n}'."
                    self._logger.info(response.message)
                    response.completions_id = n
                    request.completions_id = n
                    break
            else:
                response.success = False
                if len(self.completions) == 1:
                    response.message = "Failed to acquire completions node. The completions node is currently locked."
                else:
                    response.message = f"Failed to acquire completions node. All '{len(self.completions.keys())}' completions nodes are currently locked."
                self._logger.error(response.message)
            self.lock.release()

        elif request.action == "release":

            if request.completions_id == "":
                released_completions = []
                self.lock.acquire()
                for completions_id in self.completions:
                    if self.completions[completions_id]['locked']:
                        self.completions[completions_id]['locked'] = False
                        released_completions.append(completions_id)
                self.lock.release()
                if len(released_completions) == 0:
                    response.message = "All completions nodes are already released."
                    self._logger.debug(response.message)
                elif len(released_completions) == 1:
                    response.message = f"Released completions node '{released_completions}'."
                    self._logger.info(response.message)
                else:
                    response.message = f"Released '{len(released_completions)}' completions nodes: {released_completions}."
                    self._logger.info(response.message)

            elif request.completions_id in self.completions.keys():
                response.completions_id = request.completions_id
                self.lock.acquire()
                if self.completions[request.completions_id]['locked']:
                    self.completions[request.completions_id]['locked'] = False
                    response.message = f"Released completions node '{request.completions_id}'."
                    self._logger.info(response.message)
                else:
                    response.message = f"Completions node '{request.completions_id}' is already released."
                    self._logger.debug(response.message)
                self.lock.release()
            else:
                response.success = False
                response.message = f"Cannot release completions node '{request.completions_id}' because it does not exist."
                self._logger.error(response.message)

        elif request.action == "configure":

            if request.completions_id in self.completions.keys():
                self.lock.acquire()
                if not self.completions[request.completions_id]['locked']:
                    if acquire_style == 1:
                        response.message = f"Acquired completions node '{request.completions_id}'."
                        self.completions[request.completions_id]['locked'] = True
                        self._logger.info(response.message)
                    elif acquire_style == 2:
                        response.success = False
                        response.message = f"Cannot configure completions node '{request.completions_id}' because it has not been acquired."
                        self._logger.error(response.message)
                self.lock.release()
                if response.success:
                    if self.completions[request.completions_id]['locked']:
                        self._logger.debug(f"Forwarding parameter settings to completions node '{request.completions_id}'.")
                    else:
                        self._logger.warn(f"Forwarding parameter settings to completions node '{request.completions_id}' which has not been acquired.")
                    response.completions_id = request.completions_id
                    if len(request.parameter_names) == 0:
                        response.success, _message = self.reset_parameters(request.completions_id)
                        response.message = (f"{response.message} {_message}").lstrip()
                    else:
                        response.success, _message = self.set_parameters(request.completions_id, request.parameter_names, request.parameter_values)
                        response.message = (f"{response.message} {_message}").lstrip()
            else:
                response.success = False
                response.message = f"Cannot configure completions node '{request.completions_id}' because it does not exist."
                self._logger.error(response.message)

        else:
            response.success = False
            response.message = f"Unknown action '{request.action}'. Valid actions are 'acquire','release', and 'configure'."
            self._logger.error(response.message)

        return response

    def get_status(self, request, response):
        response.success = True

        self.lock.acquire()
        for n in self.parameters.managed_nodes:
            if n == "":
                continue
            response.completions_id.append(n)
            response.acquired.append(self.completions[n]['locked'])
        self.lock.release()

        if len(response.completions_id) == 1:
            if response.acquired[-1]:
                response.message = "The completions node is currently acquired."
            else:
                response.message = "The completions node is not currently acquired."
        else:
            num_acquired = sum(response.acquired)
            if num_acquired == len(response.completions_id):
                response.message = f"All '{len(response.completions_id)}' completions nodes are currently acquired."
            elif num_acquired == 1:
                response.message = f"'{num_acquired}' of '{len(response.completions_id)}' completions nodes is currently acquired."
            elif num_acquired > 1:
                response.message = f"'{num_acquired}' of '{len(response.completions_id)}' completions nodes are currently acquired."
            else:
                response.message = f"None of the '{len(response.completions_id)}' completions nodes are currently acquired."

        return response

    # Model Parameters

    def get_completions_settings(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward request to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self._logger.info(response.message)
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward parameter-retrieval request to completions node '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding parameter-retrieval request to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding parameter-retrieval request to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['get_parameters'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_parameters'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        rcl_request = GetParameters.Request()
                        rcl_request.names = list(self.valid_completions_parameters.keys())
                        for name in self.completions_parameters_exclude_from_get:
                            rcl_request.names.remove(name)
                        self._logger.debug(f"Request: {rcl_request}")
                        try:
                            future = self.completions[request.completions_id]['get_parameters'].call_async(rcl_request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                            if future.done():
                                rcl_response = future.result()
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['get_parameters'].srv_name}': {rcl_response}")
                                try:
                                    parameter_names = []
                                    parameter_types = []
                                    parameter_values = []

                                    if not len(rcl_request.names) == len(rcl_response.values):
                                        raise Exception(f"Expected number of received parameter values '{len(rcl_response.values)}' to to match number of sent parameter names '{len(rcl_request.names)}'")

                                    for i, p in enumerate(rcl_response.values):
                                        parameter_names.append(rcl_request.names[i])
                                        if not p.type == self.valid_completions_parameters[rcl_request.names[i]]:
                                            raise Exception(f"Expected parameter '{rcl_request.names[i]}' to be of type '{self.valid_completions_parameters[rcl_request.names[i]]}' instead of '{p.type}'")
                                        parameter_types.append(p.type)

                                        if p.type == ParameterType.PARAMETER_BOOL:
                                            parameter_values.append(str(p.bool_value))
                                        elif p.type == ParameterType.PARAMETER_INTEGER:
                                            parameter_values.append(str(p.integer_value))
                                        elif p.type == ParameterType.PARAMETER_DOUBLE:
                                            parameter_values.append(str(p.double_value))
                                        elif p.type == ParameterType.PARAMETER_STRING:
                                            parameter_values.append(p.string_value)
                                        elif p.type == ParameterType.PARAMETER_STRING_ARRAY:
                                            parameter_values.append(json.dumps(p.string_array_value))
                                        else:
                                            raise Exception(f"Parameter type '{p.type}' not implemented.")

                                except Exception as e:
                                    response.success = False
                                    response.message = (response.message + f" Failed to parse response: {e}").lstrip()
                                else:
                                    response.success = True
                                    response.message = (response.message + f" Retrieved parameters of completions node '{request.completions_id}'.").lstrip()
                                    response.parameter_names = parameter_names
                                    response.parameter_types = parameter_types
                                    response.parameter_values = parameter_values
                            else:
                                self.completions[request.completions_id]['get_parameters'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_parameters'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward request to completions node '{request.completions_id}': {repr(e)}").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    def set_parameters(self, completions, names, values):
        if len(names) != len(values):
            success = False
            message = f"Cannot configure completions node '{completions}' because the number of provided parameter names '{len(names)}' and values '{len(values)}' does not match."
            self._logger.error(message)
            return success, message

        for i, name in enumerate(names):
            if name not in self.valid_completions_parameters.keys():
                success = False
                message = f"Failed to set parameters of completions node '{completions}' because parameter '{name}' does not exist."
                self._logger.error(message)
                return success, message

        try:
            available = self.completions[completions]['set_parameters'].wait_for_service(timeout_sec=self.parameters.timeout_service)
        except KeyboardInterrupt:
            raise SelfShutdown
        else:
            if available:
                try:
                    request = SetParametersAtomically.Request()
                    for name, value in zip(names, values):
                        if self.valid_completions_parameters[name] == ParameterType.PARAMETER_DOUBLE:
                            parameter = rclpy.parameter.Parameter(name=name, value=float(value))
                        elif self.valid_completions_parameters[name] == ParameterType.PARAMETER_INTEGER:
                            parameter = rclpy.parameter.Parameter(name=name, value=int(value))
                        elif self.valid_completions_parameters[name] == ParameterType.PARAMETER_BOOL:
                            parameter = rclpy.parameter.Parameter(name=name, value=value.lower() == "true")
                        elif self.valid_completions_parameters[name] == ParameterType.PARAMETER_STRING:
                            parameter = rclpy.parameter.Parameter(name=name, value=value)
                        elif self.valid_completions_parameters[name] == ParameterType.PARAMETER_STRING_ARRAY:
                            parameter = rclpy.parameter.Parameter(name=name, value=json.loads(value), type_=list(rclpy.parameter.Parameter.Type)[self.valid_completions_parameters[name]])
                        else:
                            raise Exception(f"Parameter type '{self.valid_completions_parameters[name]}' not implemented.")
                        request.parameters.append(parameter.to_parameter_msg())
                    self._logger.debug(f"Request: {request}")
                    future = self.completions[completions]['set_parameters'].call_async(request)
                    block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                    if future.done():
                        response = future.result()
                        self._logger.debug(f"Received response from completions node '{self.completions[completions]['set_parameters'].srv_name}': {response}")
                        success = response.result.successful
                        if success:
                            if response.result.reason == "":
                                message = f"Set parameters of completions node '{completions}'."
                            else:
                                message = f"Set parameters of completions node '{completions}': {response.result.reason}"
                            self._logger.debug(message)
                        else:
                            if response.result.reason == "":
                                message = f"Failed to set parameters of completions node '{completions}'."
                            else:
                                message = f"Failed to set parameters of completions node '{completions}': {response.result.reason}"
                            self._logger.error(f"Failed to set parameters of completions node '{completions}'.")
                    else:
                        success = False
                        message = f"Failed to set parameters of completions node '{completions}' because the service '{self.completions[completions]['set_parameters'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'."
                        self.completions[completions]['set_parameters'].remove_pending_request(future)
                        self._logger.error(message)
                except Exception as e:
                    success = False
                    message = f"Error occurred while configuring parameters of completions node '{completions}': {repr(e)}"
                    self._logger.error(message)
                except KeyboardInterrupt:
                    raise SelfShutdown
            else:
                success = False
                message = f"Failed to set parameters of completions node '{completions}' because the service '{self.completions[completions]['set_parameters'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'."
                self._logger.error(message)

        return success, message

    def reset_parameters(self, completions):
        try:
            available = self.completions[completions]['reset'].wait_for_service(timeout_sec=self.parameters.timeout_service)
        except KeyboardInterrupt:
            raise SelfShutdown
        else:
            if not available:
                success = False
                message = f"Cannot reset parameters of completions node '{completions}' because the service '{self.completions[completions]['reset'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'."
                self._logger.error(message)
            else:
                try:
                    future = self.completions[completions]['reset'].call_async(TriggerFeedback.Request())
                    block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                    if future.done():
                        response = future.result()
                        self._logger.debug(f"Received response from completions node '{self.completions[completions]['reset'].srv_name}': {response}")
                        success = response.success
                        if success:
                            message = f"Reset parameters of completions node '{completions}' to default values."
                            self._logger.debug(message)
                        else:
                            if response.message == "":
                                message = f"Failed to reset parameters of completions node '{completions}' to default values."
                            else:
                                message = f"Failed to reset parameters of completions node '{completions}' to default values: {response.message}"
                            self._logger.error(f"Failed to reset parameters of completions node '{completions}' to default values.")
                    else:
                        self.self.completions[completions]['reset'].remove_pending_request(future)
                        success = False
                        message = f"Cannot reset parameters of completions node '{completions}' because the service '{self.completions[completions]['reset'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'."
                        self._logger.error(message)
                except Exception as e:
                    success = False
                    message = f"Failed to reset parameters of completions node '{completions}': {repr(e)}"
                    self._logger.error(message)
                except KeyboardInterrupt:
                    raise SelfShutdown

        return success, message

    # Prompting

    def forward_prompt(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward prompt to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                    self._logger.info(response.message)
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward interrupt request to completions node '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding prompt to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding prompt to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['prompt'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward prompt to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['prompt'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['prompt'].call_async(request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_completion + self.parameters.timeout_service)
                            if future.done():
                                log_error = False
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['prompt'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['prompt'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward prompt to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['prompt'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_completion + self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward prompt to completions node '{request.completions_id}': {repr(e)}").lstrip()
                            self._logger.error(f"Failed to forward prompt to completions node '{request.completions_id}': {repr(e)}")
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    def forward_interrupt(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward interrupt request to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                    self._logger.info(response.message)
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward interrupt request to completions node '{request.completions_id}' because it has not been acquired."
                    self._logger.error(response.message)
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding interrupt request to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding interrupt request to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['interrupt'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward interrupt request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['interrupt'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['interrupt'].call_async(request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                            if future.done():
                                log_error = False
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['interrupt'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['interrupt'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward interrupt request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['interrupt'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward interrupt request to completions node '{request.completions_id}': {repr(e)}").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    def forward_get_tools(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward tool-retrieval request to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                    self._logger.info(response.message)
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward tool-retrieval request to completions node '{request.completions_id}' because it has not been acquired."
                    self._logger.error(response.message)
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding tool-retrieval request to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding tool-retrieval request to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['get_tools'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward tool-retrieval request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_tools'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['get_tools'].call_async(request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                            if future.done():
                                log_error = False
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['get_tools'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['get_tools'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward tool-retrieval request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_tools'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward tool-retrieval request to completions node '{request.completions_id}': {repr(e)}").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    def forward_set_tools(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward tool-update request to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                    self._logger.info(response.message)
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward tool-update request to completions node '{request.completions_id}' because it has not been acquired."
                    self._logger.error(response.message)
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding tool-update request to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding tool-update request to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['set_tools'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward tool-update request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['set_tools'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['set_tools'].call_async(request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                            if future.done():
                                log_error = False
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['set_tools'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['set_tools'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward tool-update request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['set_tools'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward tool-update request to completions node '{request.completions_id}': {repr(e)}").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    # Message History

    def forward_get_context(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward context-retrieval request to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                    self._logger.info(response.message)
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward context-retrieval request to completions node '{request.completions_id}' because it has not been acquired."
                    self._logger.error(response.message)
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding context-retrieval request to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding context-retrieval request to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['get_context'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward context-retrieval request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_context'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['get_context'].call_async(request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                            if future.done():
                                log_error = False
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['get_context'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['get_context'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward context-retrieval request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_context'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward context-retrieval request to completions node '{request.completions_id}': {repr(e)}").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    def forward_set_context(self, request, response):
        response.success = True
        response.message = ""
        log_error = True

        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward context-update request to completions node '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions node '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                    self._logger.info(response.message)
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward context-update request to completions node '{request.completions_id}' because it has not been acquired."
                    self._logger.error(response.message)
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self._logger.debug(f"Forwarding context-update request to completions node '{request.completions_id}'.")
                else:
                    self._logger.warn(f"Forwarding context-update request to completions node '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['set_context'].wait_for_service(timeout_sec=self.parameters.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward context-update request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['set_context'].srv_name}' is not available: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['set_context'].call_async(request)
                            block_until_future_complete(self, future, timeout=self.parameters.timeout_service)
                            if future.done():
                                log_error = False
                                self._logger.debug(f"Received response from completions node '{self.completions[request.completions_id]['set_context'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['set_context'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward context-update request to completions node '{request.completions_id}' because the service '{self.completions[request.completions_id]['set_context'].srv_name}' is not responding: Timeout after '{self.parameters.timeout_service}s'.").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward context-update request to completions node '{request.completions_id}': {repr(e)}").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        if log_error and not response.success:
            self._logger.error(response.message)

        return response

    # Completions Status

    def publish_status(self):
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK # OK, WARN, ERROR, STALE
        status.name = self.node_name
        status.message = "status"
        status.hardware_id = "tts"

        kv = KeyValue()
        kv.key = "Stamp (seconds.nanoseconds)"
        now = self.get_clock().now().seconds_nanoseconds()
        kv.value = f"{now[0]}.{now[1]}"
        status.values.append(kv)

        names = list(self.completions.keys())

        kv = KeyValue()
        kv.key = "managed"
        kv.value = f"{len(names)}"
        status.values.append(kv)

        kv = KeyValue()
        kv.key = "names"
        kv.value = f"{names}"
        status.values.append(kv)

        locked = []
        self.lock.acquire()
        for n in names:
            locked.append(self.completions[n]['locked'])
        self.lock.release()

        kv = KeyValue()
        kv.key = "locked"
        kv.value = f"{locked}"
        status.values.append(kv)

        self.pub_status.publish(status)

def main(args=None):
    start_and_spin_node(CompletionsMultiplexer, args=args)

if __name__ == '__main__':
    main()
