#!/usr/bin/env python3

import threading

import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange, FloatingPointRange

from nimbro_api.utils.parameter_handler import ParameterHandler
from nimbro_api.utils.node import block_until_future_complete, start_and_spin_node, SelfShutdown
from nimbro_api_interfaces.srv import CompletionsManage, CompletionsGetStatus, CompletionsGetSettings
from nimbro_api_interfaces.srv import CompletionsPrompt, CompletionsStop, CompletionsGetTools, CompletionsSetTools, CompletionsGetContext, CompletionsRemoveContext, TriggerFeedback

### <Parameter Defaults>

node_name = "completions_multiplexer"
log_level = 10
managed_nodes = []

timeout_service = 5.0 # seconds
timeout_completion = 300.0 # seconds

## non-params

acquire_style = 1 # When forwarding to completions node that is not acquired: Just warn (0), Acquire (1), Block (2)
status_interval = 1.0 # seconds

### </Parameter Defaults>

class CompletionsMultiplexer(Node):

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
        descriptor.name = "managed_nodes"
        descriptor.type = ParameterType.PARAMETER_STRING_ARRAY
        descriptor.description = "Names of the LLM nodes to be managed."
        descriptor.read_only = True
        try:
            descriptor.dynamic_typing = True
        except AttributeError:
            pass
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, managed_nodes, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "timeout_completion"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Time in seconds since chat completion call after which chat completion gets aborted to forward a invalid response."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = 0.5
        float_range.to_value = 1000.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, timeout_completion, descriptor)

        descriptor = ParameterDescriptor()
        descriptor.name = "timeout_service"
        descriptor.type = ParameterType.PARAMETER_DOUBLE
        descriptor.description = "Time in seconds after which service requests timeout that should respond immediately."
        descriptor.read_only = False
        float_range = FloatingPointRange()
        float_range.from_value = 0.5
        float_range.to_value = 100.0
        float_range.step = 0.0
        descriptor.floating_point_range.append(float_range)
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, timeout_service, descriptor)

        self.parameter_handler.all_declared()

        self.completions = {}
        for n in self.managed_nodes:
            self.completions[n] = {}
            self.completions[n]['locked'] = False

            cbg_prompt = MutuallyExclusiveCallbackGroup()

            self.completions[n]['prompt'] = self.create_client(CompletionsPrompt, f"/{n}/prompt".replace("//", "/"), callback_group=cbg_prompt)
            self.completions[n]['stop'] = self.create_client(CompletionsStop, f"/{n}/stop".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())

            self.completions[n]['get_tools'] = self.create_client(CompletionsGetTools, f"/{n}/get_tools".replace("//", "/"), callback_group=cbg_prompt)
            self.completions[n]['set_tools'] = self.create_client(CompletionsSetTools, f"/{n}/set_tools".replace("//", "/"), callback_group=cbg_prompt)

            self.completions[n]['get_context'] = self.create_client(CompletionsGetContext, f"/{n}/get_context".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())
            self.completions[n]['remove_context'] = self.create_client(CompletionsRemoveContext, f"/{n}/remove_context".replace("//", "/"), callback_group=cbg_prompt)

            self.completions[n]['get_parameters'] = self.create_client(GetParameters, f"/{n}/get_parameters".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())
            self.completions[n]['set_parameters'] = self.create_client(SetParameters, f"/{n}/set_parameters".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())
            self.completions[n]['reset'] = self.create_client(TriggerFeedback, f"/{n}/reset_parameters".replace("//", "/"), callback_group=MutuallyExclusiveCallbackGroup())

        self.valid_completions_parameters = {
            'logger_level': ParameterType.PARAMETER_INTEGER,
            'probe_api_connection': ParameterType.PARAMETER_BOOL,
            'api_endpoint': ParameterType.PARAMETER_STRING,
            'model_name': ParameterType.PARAMETER_STRING,
            'model_temperatur': ParameterType.PARAMETER_DOUBLE,
            'model_top_p': ParameterType.PARAMETER_DOUBLE,
            'model_max_tokens': ParameterType.PARAMETER_INTEGER,
            'model_presence_penalty': ParameterType.PARAMETER_DOUBLE,
            'model_frequency_penalty': ParameterType.PARAMETER_DOUBLE,
            'stream_completion': ParameterType.PARAMETER_BOOL,
            'normalize_text_response': ParameterType.PARAMETER_BOOL,
            'max_tool_calls_per_response': ParameterType.PARAMETER_INTEGER,
            'correction_attempts': ParameterType.PARAMETER_INTEGER,
            'timeout_chunk': ParameterType.PARAMETER_DOUBLE,
            'timeout_completion': ParameterType.PARAMETER_DOUBLE
        }

        self.lock = threading.Lock()

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=50)

        self.srv_manage = self.create_service(CompletionsManage, f"{self.node_namespace}/{self.node_name}/manage".replace("//", "/"), self.manage_completions, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_status = self.create_service(CompletionsGetStatus, f"{self.node_namespace}/{self.node_name}/get_status".replace("//", "/"), self.get_status, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_settings = self.create_service(CompletionsGetSettings, f"{self.node_namespace}/{self.node_name}/get_settings".replace("//", "/"), self.get_completions_settings, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())

        self.srv_prompt = self.create_service(CompletionsPrompt, f"{self.node_namespace}/{self.node_name}/prompt".replace("//", "/"), self.forward_prompt, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_stop = self.create_service(CompletionsStop, f"{self.node_namespace}/{self.node_name}/stop".replace("//", "/"), self.forward_stop, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_tools = self.create_service(CompletionsGetTools, f"{self.node_namespace}/{self.node_name}/get_tools".replace("//", "/"), self.forward_get_tools, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_set_tools = self.create_service(CompletionsSetTools, f"{self.node_namespace}/{self.node_name}/set_tools".replace("//", "/"), self.forward_set_tools, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_get_context = self.create_service(CompletionsGetContext, f"{self.node_namespace}/{self.node_name}/get_context".replace("//", "/"), self.forward_get_context, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())
        self.srv_remove_context = self.create_service(CompletionsRemoveContext, f"{self.node_namespace}/{self.node_name}/remove_context".replace("//", "/"), self.forward_remove_context, qos_profile=qos_profile, callback_group=ReentrantCallbackGroup())

        self.pub_status = self.create_publisher(DiagnosticStatus, f"{self.node_namespace}/{self.node_name}/status".replace("//", "/"), qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.timer_status = self.create_timer(status_interval, self.publish_status, callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info("Node started")

    def __del__(self):
        self.get_logger().info("Node shutdown")

    def parameter_changed(self, parameter):
        success = True
        reason = ""

        if parameter.name == "logger_level":
            rclpy.logging.set_logger_level(f"{self.node_namespace}/{self.node_name}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(parameter.value))

        elif parameter.name == "managed_nodes":
            self.managed_nodes = parameter.value

        elif parameter.name == "timeout_completion":
            self.timeout_completion = parameter.value

        elif parameter.name == "timeout_service":
            self.timeout_service = parameter.value

        else:
            return None, None

        return success, reason

    # Completions Allocation

    def manage_completions(self, request, response):
        response.success = True
        response.message = ""
        response.completions_id = ""

        if request.action == "acquire":

            if request.completions_id != "":
                self.get_logger().warn(f"Non-empty field completions_id '{request.completions_id}' is being ignored while acquiring completions.")
                request.completions_id = ""

            self.lock.acquire()
            for n in self.managed_nodes:
                if not self.completions[n]['locked']:
                    self.completions[n]['locked'] = True
                    response.message = f"Acquired completions '{n}'."
                    response.completions_id = n
                    request.completions_id = n
                    break
            else:
                response.success = False
                response.message = f"Failed to acquire completions. All '{len(self.completions.keys())}' completions are currently locked."
            self.lock.release()

        elif request.action == "release":

            if request.completions_id in self.completions.keys():
                response.completions_id = request.completions_id
                self.lock.acquire()
                if self.completions[request.completions_id]['locked']:
                    self.completions[request.completions_id]['locked'] = False
                    response.message = f"Released completions '{request.completions_id}'."
                else:
                    response.message = f"Completions '{request.completions_id}' is already released."
                self.lock.release()
            else:
                response.success = False
                response.message = f"Cannot release completions '{request.completions_id}' because it does not exist."

        elif request.action == "configure":

            if request.completions_id in self.completions.keys():
                self.lock.acquire()
                if not self.completions[request.completions_id]['locked']:
                    if acquire_style == 1:
                        response.message = f"Acquired completions '{request.completions_id}'."
                        self.completions[request.completions_id]['locked'] = True
                    elif acquire_style == 2:
                        response.success = False
                        response.message = f"Cannot configure completions '{request.completions_id}' because it has not been acquired."
                self.lock.release()
                if response.success:
                    if self.completions[request.completions_id]['locked']:
                        self.get_logger().info(f"Forwarding parameter settings to completions '{request.completions_id}'.")
                    else:
                        self.get_logger().warn(f"Forwarding parameter settings to completions '{request.completions_id}' which has not been acquired.")
                    response.completions_id = request.completions_id
                    if len(request.parameter_names) == 0:
                        success, message = self.reset_parameters(request.completions_id)
                        if success:
                            response.message = (response.message + f" Reset all parameters of completions '{request.completions_id}' to default values.").lstrip()
                        else:
                            response.success = False
                            response.message = (response.message + f" Failed to reset all parameters of completions '{request.completions_id}' to default values ({message}).").lstrip()
                    else:
                        response.success, message = self.set_parameters(request.completions_id, request.parameter_names, request.parameter_values)
                        response.message = (response.message + " " + message).lstrip()
            else:
                response.success = False
                response.message = f"Cannot configure completions '{request.completions_id}' because it does not exist."

        else:

            response.success = False
            response.message = f"Unknown action '{request.action}'. Valid actions are 'acquire','release', and 'configure'."

        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().error(response.message)

        return response

    def get_status(self, request, response):
        response.success = True
        response.message = ""

        self.lock.acquire()
        for n in self.managed_nodes:
            response.completions_id.append(n)
            response.acquired.append(self.completions[n]['locked'])
        self.lock.release()

        return response

    # Model Parameters

    def get_completions_settings(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward request to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward parameter retrieval request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding parameter retrieval request to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding parameter retrieval request to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['get_parameters'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_parameters'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        rcl_request = GetParameters.Request()
                        rcl_request.names = list(self.valid_completions_parameters.keys())
                        try:
                            future = self.completions[request.completions_id]['get_parameters'].call_async(rcl_request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['get_parameters'].srv_name}'")
                                rcl_response = future.result()
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
                                        else:
                                            raise Exception(f"Parameter type '{p.type}' not implemented")

                                except Exception as e:
                                    response.success = False
                                    response.message = (response.message + f" Failed to parse response ({e}).").lstrip()
                                else:
                                    response.success = True
                                    response.message = (response.message + f" Successfully retrieved parameters of completions '{request.completions_id}'.").lstrip()
                                    response.parameter_names = parameter_names
                                    response.parameter_types = parameter_types
                                    response.parameter_values = parameter_values
                            else:
                                self.completions[request.completions_id]['get_parameters'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_parameters'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward request to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        return response

    def set_parameters(self, completions, names, values):
        if len(names) != len(values):
            success = False
            message = f" Cannot configure completions '{completions}' because the number of provided parameter names ({len(names)}) and values ({len(values)}) does not match."
            return success, message

        success = True

        pops = []
        message = ""
        for i, name in enumerate(names):
            if name not in self.valid_completions_parameters.keys():
                pops.append(i)
                success = False
                message = (message + f" Failed to configure parameters '{names}' of completions '{completions}' to values '{values}' because parameter '{name}' does not exist.").lstrip()
            else:
                if self.valid_completions_parameters[name] == ParameterType.PARAMETER_DOUBLE:
                    values[i] = float(values[i])
                elif self.valid_completions_parameters[name] == ParameterType.PARAMETER_INTEGER:
                    values[i] = int(values[i])
                elif self.valid_completions_parameters[name] == ParameterType.PARAMETER_BOOL:
                    values[i] = values[i].lower() == "true"
        for i in sorted(pops, reverse=True):
            del names[i]
            del values[i]

        if len(names) == 0:
            return success, message

        try:
            available = self.completions[completions]['set_parameters'].wait_for_service(timeout_sec=self.timeout_service)
        except KeyboardInterrupt:
            raise SelfShutdown
        else:
            if not available:
                success = False
                message = (message + f" Failed to configure parameters {names} of completions '{completions}' to values {values} because the service '{self.completions[completions]['set_parameters'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
            else:
                try:
                    req = SetParameters.Request()
                    for name, value in zip(names, values):
                        req.parameters.append(rclpy.parameter.Parameter(name=name, value=value).to_parameter_msg())
                    future = self.completions[completions]['set_parameters'].call_async(req)
                    block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                    if future.done():
                        self.get_logger().debug(f"Received response from '{self.completions[completions]['set_parameters'].srv_name}'")
                        response = future.result()
                        successes = dict(zip(names, values))
                        for i in range(len(response.results)):
                            if not response.results[i].successful:
                                del successes[names[i]]
                                if response.results[i].reason == "":
                                    reason = f" Failed to configure parameter '{names[i]}' of completions '{completions}' to value '{values[i]}'."
                                else:
                                    reason = f" Failed to configure parameter '{names[i]}' of completions '{completions}' to value '{values[i]}' ({response.results[i].reason})."
                                success = False
                                message = (message + reason).lstrip()
                        if len(successes) > 0:
                            message = (message + f" Successfully set parameters of completions '{completions}': {successes}.").lstrip()
                    else:
                        success = False
                        message = (message + f" Failed to configure parameters {names} of completions '{completions}' to values {values} because the service '{self.completions[completions]['set_parameters'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        self.completions[completions]['set_parameters'].remove_pending_request(future)
                except Exception as e:
                    success = False
                    message = (message + f" Error occurred while configuring parameters {names} of completions '{completions}' to values {values} ({e}).").lstrip()
                except KeyboardInterrupt:
                    raise SelfShutdown

        return success, message

    def reset_parameters(self, completions):
        try:
            available = self.completions[completions]['reset'].wait_for_service(timeout_sec=self.timeout_service)
        except KeyboardInterrupt:
            raise SelfShutdown
        else:
            if not available:
                success = False
                message = f"Cannot reset parameters of completions '{completions}' because the service '{self.completions[completions]['reset'].srv_name}' is not available (Timeout after '{self.timeout_service}s')."
            else:
                try:
                    future = self.completions[completions]['reset'].call_async(TriggerFeedback.Request())
                    block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                    if future.done():
                        self.get_logger().debug(f"Received response from '{self.completions[completions]['reset'].srv_name}'")
                        response = future.result()
                        success = response.success
                        message = response.message
                    else:
                        self.self.completions[completions]['reset'].remove_pending_request(future)
                        success = False
                        message = f"Cannot reset parameters of completions '{completions}' because the service '{self.completions[completions]['reset'].srv_name}' is not responding (Timeout after '{self.timeout_service}s')."
                except Exception as e:
                    success = False
                    message = f"Failed to reset parameters of completions '{completions}' ({e})."
                except KeyboardInterrupt:
                    raise SelfShutdown

        return success, message

    # Prompting

    def forward_prompt(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward prompt to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward stop request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding prompt to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding prompt to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['prompt'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward prompt to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['prompt'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['prompt'].call_async(request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_completion + self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['prompt'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['prompt'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward prompt to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['prompt'].srv_name}' is not responding (Timeout after '{self.timeout_completion + self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward prompt to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        return response

    def forward_stop(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward stop request to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward stop request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding stop request to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding stop request to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['stop'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward stop request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['stop'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['stop'].call_async(request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['stop'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['stop'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward stop request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['stop'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward stop request to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        return response

    def forward_get_tools(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward tool retrieval request to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward tool retrieval request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding tool retrieval request to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding tool retrieval request to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['get_tools'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward tool retrieval request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_tools'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['get_tools'].call_async(request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['get_tools'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['get_tools'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward tool retrieval request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_tools'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward tool retrieval request to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        return response

    def forward_set_tools(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward tool update request to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward tool update request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding tool update request to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding tool update request to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['set_tools'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward tool update request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['set_tools'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['set_tools'].call_async(request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['set_tools'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['set_tools'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward tool update request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['set_tools'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward tool update request to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        return response

    # Message History

    def forward_get_context(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward message retrieval request to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward message retrieval request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding message retrieval request to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding message retrieval request to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['get_context'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward message retrieval request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_context'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['get_context'].call_async(request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['get_context'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['get_context'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward message retrieval request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['get_context'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward message retrieval request to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

        return response

    def forward_remove_context(self, request, response):
        response.success = True
        response.message = ""
        if request.completions_id not in self.completions.keys():
            response.success = False
            response.message = f"Cannot forward message removal request to completions '{request.completions_id}' because it does not exist."
        else:
            self.lock.acquire()
            if not self.completions[request.completions_id]['locked']:
                if acquire_style == 1:
                    response.message = f"Acquired completions '{request.completions_id}'."
                    self.completions[request.completions_id]['locked'] = True
                elif acquire_style == 2:
                    response.success = False
                    response.message = f"Cannot forward message removal request to completions '{request.completions_id}' because it has not been acquired."
            self.lock.release()
            if response.success:
                if self.completions[request.completions_id]['locked']:
                    self.get_logger().info(f"Forwarding message removal request to completions '{request.completions_id}'.")
                else:
                    self.get_logger().warn(f"Forwarding message removal request to completions '{request.completions_id}' which has not been acquired.")

                try:
                    available = self.completions[request.completions_id]['remove_context'].wait_for_service(timeout_sec=self.timeout_service)
                except KeyboardInterrupt:
                    raise SelfShutdown
                else:
                    if not available:
                        response.success = False
                        response.message = (response.message + f" Cannot forward message removal request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['remove_context'].srv_name}' is not available (Timeout after '{self.timeout_service}s').").lstrip()
                    else:
                        try:
                            future = self.completions[request.completions_id]['remove_context'].call_async(request)
                            block_until_future_complete(self, future, timeout_sec=self.timeout_service)
                            if future.done():
                                self.get_logger().debug(f"Received response from '{self.completions[request.completions_id]['remove_context'].srv_name}'")
                                response = future.result()
                            else:
                                self.completions[request.completions_id]['remove_context'].remove_pending_request(future)
                                response.success = False
                                response.message = (response.message + f" Cannot forward message removal request to completions '{request.completions_id}' because the service '{self.completions[request.completions_id]['remove_context'].srv_name}' is not responding (Timeout after '{self.timeout_service}s').").lstrip()
                        except Exception as e:
                            response.success = False
                            response.message = (response.message + f" Failed to forward message removal request to completions '{request.completions_id}' ({e}).").lstrip()
                        except KeyboardInterrupt:
                            raise SelfShutdown

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
        kv.key = "completions"
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
