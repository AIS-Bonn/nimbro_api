#!/usr/bin/env python3

import json
import traceback

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nimbro_api.api_director import ApiDirector
from nimbro_api.utils.node import start_and_spin_node

class TestNode(Node):
    def __init__(self):
        super().__init__("nimbro_api_tests")
        self.api_director = ApiDirector(self)
        self.timer_state = self.create_timer(0.0, self.state_machine, callback_group=MutuallyExclusiveCallbackGroup())
        rclpy.logging.set_logger_level(f"{self.get_namespace()}/{self.get_name()}".replace("//", "/")[1:].replace("/", "."), rclpy.logging.LoggingSeverity(10))
        self.get_logger().info("Node started")

    def state_machine(self):
        self.timer_state.cancel()
        self.get_logger().info("Doing tests")
        try:
            self.test()
        except Exception:
            self.get_logger().error(str(traceback.format_exc()))
        else:
            self.get_logger().info("Passed all tests")

    def test(self):
        parameter_sets = [
            # {
            #     'logger_level': "10",
            #     # 'probe_api_connection': "False",
            #     'stream_completion': "False",
            #     # 'api_endpoint': json.dumps({
            #     #     'name': "OpenAI",
            #     #     'api_flavor': "openai",
            #     #     'models_url': "https://api.openai.com/v1/models",
            #     #     'completions_url': "https://api.openai.com/v1/chat/completions",
            #     #     'key_type': "plain", # "environment"
            #     #     'key_value': "yourkey123"
            #     # }),
            #     'api_endpoint': "OpenAI",
            #     'model_name': "gpt-4o",
            #     # 'api_endpoint': "Mistral AI",
            #     # 'model_name': "mistral-large-latest",
            #     # 'api_endpoint': "OpenRouter",
            #     # 'model_name': "google/gemini-2.0-flash-001",
            #     # 'api_endpoint': "AIS",
            #     # 'model_name': "ais/code-llm",
            #     # 'api_endpoint': "vLLM",
            #     # 'model_name': "ais/ministral-8b",
            #     # 'model_temperatur': "0.7",
            #     # 'model_top_p': "1.0",
            #     # 'model_max_tokens': "1000",
            #     # 'model_presence_penalty': "0.0",
            #     # 'model_frequency_penalty': "0.0",
            #     # 'normalize_text_response': "False",
            #     # 'max_tool_calls_per_response': "1",
            #     'correction_attempts': "0",
            #     # 'timeout_chunk': "5.0",
            #     # 'timeout_completion': "30.0"
            # },
            {
                'logger_level': "10",
                'stream_completion': "False",
                'api_endpoint': "OpenAI",
                'model_name': "gpt-4o"
            },
            {
                'logger_level': "10",
                'stream_completion': "True",
                'api_endpoint': "OpenAI",
                'model_name': "gpt-4o"
            },
            {
                'logger_level': "10",
                'stream_completion': "False",
                'api_endpoint': "Mistral AI",
                'model_name': "mistral-large-latest"
            },
            {
                'logger_level': "10",
                'stream_completion': "True",
                'api_endpoint': "Mistral AI",
                'model_name': "mistral-large-latest"
            },
            {
                'logger_level': "10",
                'stream_completion': "False",
                'api_endpoint': "OpenRouter",
                'model_name': "google/gemini-2.0-flash-001",
            },
            {
                'logger_level': "10",
                'stream_completion': "True",
                'api_endpoint': "OpenRouter",
                'model_name': "google/gemini-2.0-flash-001",
            }
        ]

        for params in parameter_sets:
            self.get_logger().debug(f"Handling parameters: {json.dumps(params, indent=4)}")

            # release all sessions and acquire one
            if True:
                self.get_logger().debug("Starting test 'release & acquire'")

                success, message, completions_ids, completions_acquired = self.api_director.get_status()
                assert success, message
                if True not in completions_acquired:
                    self.get_logger().info("No completions have been acquired")
                else:
                    for i, completions_id in enumerate(completions_ids):
                        if completions_acquired[i]:
                            success, message = self.api_director.release(completions_id)
                            assert success, message

                success, message, completions_ids, completions_acquired = self.api_director.get_status()
                assert success, message
                assert sum(completions_acquired) == 0, f"{sum(completions_acquired)}"

                success, message, completions_id = self.api_director.acquire(reset_parameters=True, reset_context=True)
                assert success, message

                success, message, completions_ids, completions_acquired = self.api_director.get_status()
                assert success, message
                assert sum(completions_acquired) == 1, f"{sum(completions_acquired)}"

                self.get_logger().info("Passed test 'release & acquire'")

            # set model parameters
            if True:
                self.get_logger().debug("Starting test 'Set model parameters'")

                success, message = self.api_director.set_parameters(completions_id,
                                                                    list(params.keys()),
                                                                    list(params.values()))
                assert success, message

                self.get_logger().info("Passed test 'Set model parameters'")

            # basic text completion
            if True:
                self.get_logger().debug("Starting test 'Basic text completion'")

                success, message, text_response, tool_calls = self.api_director.prompt(completions_id=completions_id,
                                                                                       text='You are not allowed to tell jokes.',
                                                                                       role="system",
                                                                                       reset_context=True,
                                                                                       response_type="none")
                assert success, message

                success, message, text_response, tool_calls = self.api_director.prompt(completions_id=completions_id,
                                                                                       text='Tell me a joke about students.',
                                                                                       response_type="text")
                assert success, message

                self.get_logger().info("Passed test 'Basic text completion'")

            # complex tool usage
            if True:
                self.get_logger().debug("Starting test 'Complex tool usage'")

                tools = [
                    {
                        "name": "get_current_weather",
                        "description": "Get the current weather at the current location",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }

                    },
                    {
                        "name": "get_current_time",
                        "description": "Get the current time at the current location",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }

                    },
                    {
                        'name': "speak",
                        'description': "Speak to a person, e.g. the useself. Never use plain text repsonses to address anyone.",
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'person': {
                                    'type': "string",
                                    'description': "Specifies the person to speak to. Pass 'everyone' to address everyone in the robot's vicinity, rather than a specific person"
                                },
                                "text": {
                                    'type': "string",
                                    'description': "Specifies the text to be said. Be friendly, concise, and helpful"
                                },
                                "requires_answer": {
                                    'type': "boolean",
                                    'description': "Signals that the spoken text requires an answer and makes the robot wait for it. The answer will then be returned with the response to this function call"
                                }
                            },
                            'required': ['person', "text", "requires_answer"]
                        }
                    }
                ]
                success, message = self.api_director.set_tools(completions_id=completions_id, tools=tools)
                assert success, message

                success, message, text_response, tool_calls_a = self.api_director.prompt(completions_id=completions_id,
                                                                                         text='You are a helpful tool assistant.',
                                                                                         role='system',
                                                                                         response_type='none',
                                                                                         reset_context=True)
                assert success, message

                success, message, text_response, tool_calls = self.api_director.prompt(completions_id=completions_id,
                                                                                       text='Tell Michael how warm it is outside.',
                                                                                       # response_type='get_current_weather', # not supported by mistral
                                                                                       response_type='auto',
                                                                                       reset_context=False)
                assert success, message

                assert isinstance(tool_calls, list)
                assert len(tool_calls) == 1
                assert isinstance(tool_calls[0], dict)
                assert "id" in tool_calls[0] and "name" in tool_calls[0] and "arguments" in tool_calls[0]
                assert tool_calls[0]["name"] == "get_current_weather"

                success, message, text_response, tool_calls = self.api_director.prompt(completions_id=completions_id,
                                                                                       role="tool",
                                                                                       text='The sun is shining and it is 42°C.',
                                                                                       tool_response_id=tool_calls[0]['id'],
                                                                                       response_type='always')
                assert success, message

                assert len(tool_calls) == 1
                assert isinstance(tool_calls[0], dict)
                assert "id" in tool_calls[0] and "name" in tool_calls[0] and "arguments" in tool_calls[0]
                assert tool_calls[0]["name"] == "speak"
                assert "person" in tool_calls[0]["arguments"] and "text" in tool_calls[0]["arguments"] and "requires_answer" in tool_calls[0]["arguments"]
                assert tool_calls[0]["arguments"]["person"] == "Michael"

                success, message, text_response, tool_calls = self.api_director.prompt(completions_id=completions_id,
                                                                                       role="tool",
                                                                                       text='Ok, thank you.',
                                                                                       tool_response_id=tool_calls[0]['id'],
                                                                                       response_type='speak')

                assert success, message

                assert len(tool_calls) == 1
                assert isinstance(tool_calls[0], dict)
                assert "id" in tool_calls[0] and "name" in tool_calls[0] and "arguments" in tool_calls[0]
                assert tool_calls[0]["name"] == "speak"

                self.get_logger().info("Passed test 'Complex tool usage'")

def main(args=None):
    start_and_spin_node(TestNode, args=args)

if __name__ == '__main__':
    main()
