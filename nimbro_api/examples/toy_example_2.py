#!/usr/bin/env python3

import time
import copy
import traceback

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, IntegerRange

from nimbro_api.api_director import ApiDirector
from nimbro_api.utils.parameter_handler import ParameterHandler
from nimbro_api.utils.node import start_and_spin_node

### <Parameter Defaults>

node_name = "toy_example_2"
logger_level = 10

## non-params

attempts = 1
add_deception_functions = False

probe_api_connection = True
api_endpoint = "OpenAI" # "Mistral AI"
model_name = "gpt-4o" # "mistral-large-latest"
model_temperatur = 0.01
model_top_p = 0.01
model_max_tokens = 1000
model_presence_penalty = 0.0
model_frequency_penalty = 0.0
stream_completion = False
normalize_text_response = False
max_tool_calls_per_response = 1
correction_attempts = 3
timeout_chunk = 5.0
timeout_completion = 30.0

model_system_prompt = "You control a service robot via a set of functions in order to accomplish a given task. Carefully read the function and parameter descriptions to understand their behavior and select the appropriate ones. Expect functions to return unexpected results and adapt to it. Do not ever use the 'multi_tool_use.parallel' function. Instead, call functions sequentially, one per response."
chain_of_thought = True
chain_of_thought_texts = ["Considering all available functions, come up with a brief and conceptual step-by-step plan for accomplishing this task.", "Execute this plan by calling the appropriate functions, one per response."]
in_between_reasoning = False
in_between_reasoning_texts = ["Given all available functions, which one should you call to advance task accomplishment? Also, specify the required parameters. Keep your reasoning short, focus only on the very next function call, and conclude with a concrete a recommendation.", "Call the recommended function."]

### </Parameter Defaults>

class ToyExample(Node):

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

        self.parameter_handler.all_declared()

        self.results = []
        self.api_director = ApiDirector(self)

        self.timer_state_machine = self.create_timer(0, self.state_machine, callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info("Node started")

    def __del__(self):
        self.get_logger().info("Node shutdown")

    def parameter_changed(self, parameter):
        success = True
        reason = ""

        if parameter.name == "logger_level":
            rclpy.logging.set_logger_level(f"{self.node_namespace}/{self.node_name}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(parameter.value))

        else:
            return None, None

        return success, reason

    def configure_model(self):
        self.get_logger().debug("Configuring LLM")

        success, message, self.completions_id = self.api_director.acquire(reset_parameters=True, reset_context=True, retry=True)
        if not success:
            raise Exception(f"Unexpected failure in api_director.acquire(retry=True): {message}")
        else:
            self.get_logger().debug(f"Using completions '{self.completions_id}'")

        params_main = {
            "logger_level": str(logger_level),
            "probe_api_connection": str(probe_api_connection),
            "api_endpoint": api_endpoint,
            "model_name": model_name,
            "model_temperatur": str(model_temperatur),
            "model_top_p": str(model_top_p),
            "model_max_tokens": str(model_max_tokens),
            "model_presence_penalty": str(model_presence_penalty),
            "model_frequency_penalty": str(model_frequency_penalty),
            "stream_completion": str(stream_completion),
            "normalize_text_response": str(normalize_text_response),
            "max_tool_calls_per_response": str(max_tool_calls_per_response),
            "correction_attempts": str(correction_attempts),
            "timeout_chunk": str(timeout_chunk),
            "timeout_completion": str(timeout_completion)
        }

        success, message = self.api_director.set_parameters(self.completions_id, list(params_main.keys()), list(params_main.values()), retry=True)
        if not success:
            raise Exception(f"Unexpected failure in api_director.set_parameter(retry=True): {message}")

        self.get_logger().info("Configured LLM")

    def state_machine(self):
        self.timer_state_machine.cancel()
        try:
            self.run_example()
        except Exception:
            self.get_logger().error(str(traceback.format_exc()))

    def run_example(self):
        self.configure_model()

        ### scenario

        self.locations = ["shelve", "table", "operator", "sink"]

        self.current_location = 2
        self.objects = ["apple", "banana", "orange", "bread", "cereals", "pringles", "chocolate",
                        "coke", "orange juice", "olive oil", "red wine", "rum", "milk", "water",
                        "notebook", "rubiks cube", "lighter", "candle", "cigar", "ash tray", "pen",
                        "bowl", "plate", "fork", "knife", "spoon", "cup", "glass",
                        "sponge", "cloth", "brush", "dish soap", "cleaner spray", "paper roll", "kitchen towel"]
        self.object_locations = {"apple": 0, "banana": 0, "orange": 0, "bread": 0, "cereals": 0, "pringles": 0, "chocolate": 0,
                                 "coke": 0, "orange juice": 0, "olive oil": 0, "red wine": 0, "rum": 0, "milk": 0, "water": 0,
                                 "notebook": 0, "rubiks cube": 0, "lighter": 0, "candle": 0, "cigar": 0, "ash tray": 0, "pen": 0,
                                 "bowl": 0, "plate": 0, "fork": 0, "knife": 0, "spoon": 0, "cup": 0, "glass": 0,
                                 "sponge": 0, "cloth": 0, "brush": 0, "dish soap": 0, "cleaner spray": 0, "paper roll": 0, "kitchen towel": 0}

        # self.task = "Put the apple on the table."
        # def goal_reached():
        #   return self.object_locations["apple"] == 1

        # self.task = "Put everything I need to eat cereals on the table."
        self.task = "I want to east granola. Set the table."

        def goal_reached():
            return self.object_locations["cereals"] == 1 and self.object_locations["milk"] == 1 and self.object_locations["bowl"] == 1 and self.object_locations["spoon"] == 1

        # self.task = "Give me a cigar and fire and put the ash tray on the table."
        # def goal_reached():
        #   return self.object_locations["cigar"] == 2 and self.object_locations["lighter"] == 2 and self.object_locations["ash tray"] == 1

        # self.object_locations["cereals"] = 1
        # self.object_locations["milk"] = 1
        # self.object_locations["bowl"] = 1
        # self.object_locations["spoon"] = 1
        # self.task = "Put the dirty dishes into the sink and the ingredients back into the shelve."
        # def goal_reached():
        #   return self.object_locations["cereals"] == 0 and self.object_locations["milk"] == 0 and self.object_locations["bowl"] == 3 and self.object_locations["spoon"] == 3

        ### run

        if len(self.results) == 0:
            print()
            self.get_logger().info("Scenario:")
            self.get_logger().info("Locations: " + str(self.locations))
            self.get_logger().info("Current location: " + self.locations[self.current_location])
            self.get_logger().info("Objects & locations: " + str(self.object_locations))
            self.get_logger().info("Task: '" + self.task + "'")

        print()
        self.get_logger().info("Running attempt " + str(len(self.results) + 1) + "/" + str(attempts))

        iterations = 0
        stop_reason = ""
        tic = time.perf_counter()

        info = copy.deepcopy(self.object_locations)
        for k in self.object_locations.keys():
            info[k] = self.locations[self.object_locations[k]]

        while True:
            try:
                text_response # noqa
            except UnboundLocalError:
                self.update_tools()

                text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=model_system_prompt, role="system", reset_context=True, response_type="none", retry=True)[2:]
                text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text="Task: " + self.task, role="user", reset_context=False, response_type="none", retry=True)[2:]
                text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text="Here is additional information regarding object locations:\n" + str(info), role="user", reset_context=False, response_type="none" if chain_of_thought or in_between_reasoning else "always", retry=True)[2:]
                if chain_of_thought:
                    text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=chain_of_thought_texts[0], role="user", reset_context=False, response_type="text", retry=True)[2:]
                    if in_between_reasoning:
                        text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=in_between_reasoning_texts[0], role="user", reset_context=False, response_type="text", retry=True)[2:]
                        text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=in_between_reasoning_texts[1], role="user", reset_context=False, response_type="always", retry=True)[2:]
                    else:
                        text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=chain_of_thought_texts[1], role="user", reset_context=False, response_type="always", retry=True)[2:]
                elif in_between_reasoning:
                    text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=in_between_reasoning_texts[0], role="user", reset_context=False, response_type="text", retry=True)[2:]
                    text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=in_between_reasoning_texts[1], role="user", reset_context=False, response_type="always", retry=True)[2:]
            finally:
                if len(tool_calls) == 1:
                    tool = tool_calls[0]
                else:
                    log = "Model responded with '" + str(len(tool_calls)) + "' tool calls"
                    self.get_logger().error(log)
                    if text_response == "":
                        stop_reason = log
                    else:
                        stop_reason = text_response
                    break
            iterations += 1

            if tool["name"] == "drive_to_location":
                if tool["arguments"]["location"] in self.locations[:self.current_location] + self.locations[self.current_location + 1:]:
                    self.current_location = self.locations.index(tool["arguments"]["location"])
                    self.get_logger().info("Current location set to '" + self.locations[self.current_location] + "'")
                    self.update_tools()
                    text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text="The robot successfully moved to the location '" + self.locations[self.current_location] + "'.", role="tool", reset_context=False, tool_response_id=tool["id"], response_type="none" if in_between_reasoning else "always", retry=True)[2:]
                else:
                    stop_reason = "Location '" + tool["arguments"]["location"]["name"] + "' is not a valid target"
                    break

            elif tool["name"] == "grasp_object":
                if tool["arguments"]["object"] in self.object_locations:
                    if self.object_locations[tool["arguments"]["object"]] == self.current_location:
                        self.object_locations[tool["arguments"]["object"]] = len(self.locations)
                        self.get_logger().info("Grasped object '" + tool["arguments"]["object"] + "'")
                        self.update_tools()
                        text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text="The robot successfully grasped the object '" + tool["arguments"]["object"] + "'.", role="tool", reset_context=False, tool_response_id=tool["id"], response_type="none" if in_between_reasoning else "always", retry=True)[2:]
                    else:
                        # text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text="The " + tool["arguments"]["object"] + " is not in reach.", role="tool", reset_context=False, tool_response_id=tool["id"], response_type="none" if in_between_reasoning else "always", retry=True)[2:]
                        stop_reason = "Object '" + tool["arguments"]["object"] + "' is not in reach"
                        break
                else:
                    stop_reason = "Object '" + tool["arguments"]["object"] + "' is not a valid object"
                    break

            elif tool["name"] == "place_object":
                if tool["arguments"]["object"] in self.object_locations:
                    if self.object_locations[tool["arguments"]["object"]] == len(self.locations):
                        self.object_locations[tool["arguments"]["object"]] = self.current_location
                        self.get_logger().info("Placed object '" + tool["arguments"]["object"] + "'")
                        self.update_tools()
                        text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text="The robot successfully placed the object '" + tool["arguments"]["object"] + "' at the current location.", role="tool", reset_context=False, tool_response_id=tool["id"], response_type="none" if in_between_reasoning else "always", retry=True)[2:]
                    else:
                        stop_reason = "Object '" + tool["arguments"]["object"] + "' is not grasped"
                        break
                else:
                    stop_reason = "Object '" + tool["arguments"]["object"] + "' is not a valid object"
                    break

            elif tool["name"] == "task_is_accomplished":
                stop_reason = "Model called task_is_accomplished()"
                break

            else:
                stop_reason = "Function '" + tool["name"] + "' does not exist"
                break

            if in_between_reasoning:
                text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=in_between_reasoning_texts[0], role="user", reset_context=False, response_type="text", retry=True)[2:]
                text_response, tool_calls = self.api_director.prompt(completions_id=self.completions_id, text=in_between_reasoning_texts[1], role="user", reset_context=False, response_type="always", retry=True)[2:]

        ### evaluate

        toc = time.perf_counter()
        if stop_reason != "":
            if stop_reason == "Model called task_is_accomplished()":
                self.get_logger().info(stop_reason)
            else:
                self.get_logger().error(stop_reason)

        success = goal_reached()
        if success:
            self.get_logger().info("Attempt " + str(len(self.results) + 1) + "/" + str(attempts) + " done (goal_reached=" + str(success) + ", iterations=" + str(iterations) + ")")
        else:
            self.get_logger().warn("Attempt " + str(len(self.results) + 1) + "/" + str(attempts) + " done (goal_reached=" + str(success) + ", iterations=" + str(iterations) + ")")
        if not success:
            self.get_logger().info("Current location: " + self.locations[self.current_location])
            self.get_logger().info("Objects & locations: " + str(self.object_locations))

        self.results.append((success, iterations, stop_reason, toc - tic))

        if len(self.results) >= attempts:
            print()
            self.get_logger().info("Experiment finished")
            print()
            self.get_logger().info("Results (goal_reached, iterations, stop_reason, time): " + str(self.results))
            w = 0
            for i in range(len(self.results)):
                if self.results[i][0]:
                    w += 1
            self.get_logger().info("In total, " + str(w) + "/" + str(len(self.results)) + " attempt " + ("was" if w == 1 else "were") + " successful")

    def update_tools(self):
        tools = []

        tools.append({
            "name": "drive_to_location",
            "description": "Drive to the specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "There are " + str(len(self.locations)) + " locations in total. The robot's current location is '" + self.locations[self.current_location] + "'",
                        "enum": self.locations[:self.current_location] + self.locations[self.current_location + 1:]
                    }
                },
                "required": ["location"]
            }
        })

        tools.append({
            "name": "grasp_object",
            "description": "Grasp an object that is within reach from the current location",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "The object to be grasped",
                        "enum": [k for k, v in self.object_locations.items() if v == self.current_location]
                    }
                },
                "required": ["object"]
            }
        })

        tools.append({
            "name": "place_object",
            "description": "Place an object held in your hand at the current location",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "The object to be placed",
                        "enum": [k for k, v in self.object_locations.items() if v == len(self.locations)]
                    }
                },
                "required": ["object"]
            }
        })

        tools.append({
            "name": "task_is_accomplished",
            "description": "The given task '" + self.task + "' is fully accomplished",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })

        # tools.append({
        #       "name": "task_completed",
        #       "description": "The given task '" + self.task + "' is either fully accomplished or cannot be fully accomplished even though you tried everything",
        #       "parameters": {
        #           "type": "object",
        #           "properties": {
        #               "success": {
        #                   "type": "boolean",
        #                   "description": "'true' if the task is fully accomplished or 'false' if it cannot be fully accomplished",
        #                   "enum": [k for k, v in self.object_locations.items() if v == len(self.locations)]
        #               }
        #           },
        #           "required": ["success"]
        #       }
        #   })

        # deception

        if add_deception_functions:
            tools.append({
                "name": "do_backflip",
                "description": "Do a badass backflip at the current location",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            })

            tools.append({
                "name": "vacuum_clean",
                "description": "Vacuum clean the current location",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            })

            tools.append({
                "name": "wait_for_person",
                "description": "Wait for a person at the current location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {
                            "type": "string",
                            "description": "Person to wait for (use 'any' to not wait for a specific person)"
                        }
                    },
                    "required": ["person"]
                }
            })

            tools.append({
                "name": "high_five",
                "description": "Give a high-five to a person in arm range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {
                            "type": "string",
                            "description": "Person to give a high-five to"
                        }
                    },
                    "required": ["person"]
                }
            })

            tools.append({
                "name": "speak",
                "description": "Speak to a person in range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {
                            "type": "string",
                            "description": "Person to speak to (use 'all' for adressing everyone in range)"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to speak to the person"
                        }
                    },
                    "required": ["person", "text"]
                }
            })

            tools.append({
                "name": "receive_reply",
                "description": "Listen to or receive a reply from a person in range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {
                            "type": "string",
                            "description": "Person to listen to (use 'all' for listening to everyone in range)"
                        }
                    },
                    "required": ["person"]
                }
            })

        self.api_director.set_tools(completions_id=self.completions_id, tools=tools, retry=True)

def main(args=None):
    start_and_spin_node(ToyExample, args=args)

if __name__ == '__main__':
    main()
