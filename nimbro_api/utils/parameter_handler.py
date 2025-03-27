#!/usr/bin/env python3

import copy
import traceback

import rclpy
from rcl_interfaces.msg import SetParametersResult

# TUTORIAL
#
# 1. In your nodes __init__ function do:
#
# self.parameter_handler = ParameterHandler(self)
# self.add_on_set_parameters_callback(self.parameter_handler.parameter_callback)
# declare parameters...
# self.parameter_handler.all_declared()
#
# 2. Define as a member of your node class:
#
# def parameter_changed(self, parameter):
#     success = True
#     reason = ""
#
#    if parameter.name == "my_parameter_1":
#         self.parameter_1 = parameter.value
#
#    elif parameter.name == "my_parameter_2":
#         if parameter.value == "bad value":
#             success = False
#             reason = f"Value '{parameter.value}' is not permitted"
#         else:
#             self.parameter_2 = parameter.value
#
#     else:
#         return None, None
#
#     return success, reason

parameter_types = ["Type.NOT_SET",
                   "Type.BOOL",
                   "Type.INTEGER",
                   "Type.DOUBLE",
                   "Type.STRING",
                   "Type.BYTE_ARRAY",
                   "Type.BOOL_ARRAY",
                   "Type.INTEGER_ARRAY",
                   "Type.DOUBLE_ARRAY",
                   "Type.STRING_ARRAY"]

class ParameterHandler:

    def __init__(self, node, verbose=False, logger_prefix="[parameter_handler] "):
        assert isinstance(node, rclpy.node.Node), f"ParameterHandler requires a reference to an object of type 'rclpy.node.Node' but got a reference to an object of type '{type(node).__name__}'!"
        assert isinstance(verbose, bool), f"Expected 'verbose' to be of type 'boll' but it is of type '{type(verbose).__name__}'!"
        assert isinstance(logger_prefix, str), f"Expected 'logger_prefix' to be of type 'str' but it is of type '{type(logger_prefix).__name__}'!"

        assert not hasattr(node, "setup_finished"), f"Expected object '{node.__class__.__name__}' to not have an attribute 'setup_finished'"
        assert not hasattr(node, "parameter_descriptors"), f"Expected object '{node.__class__.__name__}' to not have an attribute 'parameter_descriptors'"

        self.node = node
        self.node.setup_finished = False
        self.node.parameter_descriptors = []
        self.verbose = verbose
        self.logger_prefix = logger_prefix

    def all_declared(self):
        self.node.setup_finished = True
        self.node.parameter_descriptors = copy.deepcopy(self.node.parameter_descriptors)
        if self.verbose:
            self.node.get_logger().debug(self.logger_prefix + "All parameters have been declared")

    def parameter_callback(self, parameters):
        if self.verbose:
            self.node.get_logger().debug(self.logger_prefix + "Received '" + str(len(parameters)) + "' parameter change" + ("s" if len(parameters) != 1 else "") + ": " + str([p.name for p in parameters]))

        result = SetParametersResult()
        result.successful = True
        result.reason = "success"

        for i in range(len(parameters)):
            this_one_successful = True

            for j in range(len(self.node.parameter_descriptors)):
                found_descriptor = False

                if parameters[i].name == self.node.parameter_descriptors[j].name:
                    if self.verbose:
                        self.node.get_logger().debug(self.logger_prefix + "Found descriptor (ID:'" + str(j) + "')")
                    found_descriptor = True

                    d = self.node.parameter_descriptors[j]

                    if d.read_only is False or self.node.setup_finished is False:

                        if d.type == parameters[i].to_parameter_msg().value._type:
                            pass # range seems to be handled somewhere else, which implies that it will not be logged

                        else:
                            this_one_successful = False
                            result.successful = False
                            result.reason = "Rejected attempt to set parameter '" + parameters[i].name + "' of type '" + parameter_types[d.type] + "' to value '" + str(parameters[i].value) + "' of different type '" + str(parameters[i].type_) + "'"
                            self.node.get_logger().error(self.logger_prefix + result.reason)

                    else:
                        this_one_successful = False
                        result.successful = False
                        result.reason = "Rejected attempt to set parameter '" + parameters[i].name + "' to value '" + str(parameters[i].value) + "' because it is defined as read only"
                        self.node.get_logger().error(self.logger_prefix + result.reason)

                    break

            if this_one_successful:
                success = None
                try:
                    success, reason = self.node.parameter_changed(parameters[i])
                except Exception as e:
                    trace = traceback.format_exc()
                    print(f"\033[91m{trace}\033[0m")
                    success = True
                    reason = "Parameter '" + parameters[i].name + "' might not be handled correctly - Exception while calling 'parameter_changed(parameter)' callback (" + str(e) + ")"
                    result.reason = reason
                    self.node.get_logger().warn(self.logger_prefix + reason)

                if success is None:
                    success = True
                    reason = "Parameter '" + parameters[i].name + "' might not be handled correctly as it is not referenced by the 'parameter_changed(parameter)' callback"
                    result.reason = reason
                    self.node.get_logger().warn(self.logger_prefix + reason)

                if success:
                    if self.node.setup_finished:
                        self.node.get_logger().info(self.logger_prefix + "Parameter '" + parameters[i].name + "' set to '" + str(parameters[i].value) + "'")
                    else:
                        self.node.get_logger().info(self.logger_prefix + "Parameter '" + parameters[i].name + "' initialized to '" + str(parameters[i].value) + "'")

                else:
                    result.successful = False
                    if self.node.setup_finished:
                        result.reason = "Rejected attempt to set parameter '" + parameters[i].name + "' to value '" + str(parameters[i].value) + "' (" + reason + ")"
                    else:
                        result.reason = "Rejected attempt to initialize parameter '" + parameters[i].name + "' to value '" + str(parameters[i].value) + "' (" + reason + ")"
                    self.node.get_logger().error(self.logger_prefix + result.reason)

            elif not found_descriptor:
                this_one_successful = False
                result.successful = False
                result.reason = "Rejected attempt to set parameter '" + parameters[i].name + "' to value '" + str(parameters[i].value) + "' because the corresponding descriptor cannot be found"
                self.node.get_logger().error(self.logger_prefix + result.reason)

        if result.successful is False and len(parameters) > 1:
            result.reason = "Failure during parameter callback with more than one ('" + str(len(parameters)) + "') parameter: Reason is ambiguous (see logs)"
        if self.verbose:
            self.node.get_logger().debug(self.logger_prefix + "Parameter change finished" + ": " + str([p.name for p in parameters]))

        if self.node.setup_finished is False and result.successful is False:
            self.node.get_logger().fatal(self.logger_prefix + "Shutting down node after rejected parameter initialization")
            raise SystemExit

        return result
