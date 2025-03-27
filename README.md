# nimbro_api

This package exposes various APIs established by OpenAI ([Chat Completions](https://platform.openai.com/docs/api-reference/chat), [Embeddings](https://platform.openai.com/docs/api-reference/embeddings), [Audio](https://platform.openai.com/docs/api-reference/audio), [Images](https://platform.openai.com/docs/api-reference/images)) and adopted by others to ROS2.

## Features

This package targets the ROS2 Foxy and Jazzy [distributions](https://docs.ros.org/en/rolling/Releases.html), but should be compatible with others as well.

It is completely Python based and requires [almost](https://github.com/AIS-Bonn/nimbro_api/blob/main/requirements.txt) no external dependencies.

It supports several flavors of the (Chat Completions & Embeddings) APIs to enable enpoints from [OpenAI](https://platform.openai.com/docs/api-reference/chat), [Mistral AI](https://docs.mistral.ai/api/#tag/chat), [OpenRouter](https://openrouter.ai/docs/api-reference/overview) and [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai) and any other provider that behaves exactly like one of them.

The Chat Completions integration provides streamed and asynchronous prompting, tool usage, setting tool choice, JSON mode, vision input (web and local), setting model parameters, and message history editing. It also provides monitoring of token usage and cost, validity checks for all inputs and outputs (including JSON schema compliance when using tools), robust timeout behavior, as well as optional text normalization and various self-correction routines in case the model output deviates from what is expected.

All other API integrations provide caching capabilities to reduce cost, bandwidth, and latency.

## Installation

Simply include this repository together with [the repository containing all required interfaces](https://github.com/AIS-Bonn/nimbro_api_interfaces) in the source folder of your colcon workspace. After building them

`colcon build --packages-select nimbro_api nimbro_api_interfaces`

and re-sourcing

`source install/local_setup.bash`

several [launch files](https://github.com/AIS-Bonn/nimbro_api/tree/main/launch), [nodes](https://github.com/AIS-Bonn/nimbro_api/tree/main/nimbro_api/) and [demos](https://github.com/AIS-Bonn/nimbro_api/tree/main/nimbro_api/examples) will be available in your environment.

Alternatively, you may also use the provided [devcontainer](https://github.com/AIS-Bonn/nimbro_api/tree/main/.devcontainer).

## Usage

For a step-by-step guide with detailed explanations and extensive examples on how to use this package, see the Jupyter Notebook [tutorial](https://github.com/AIS-Bonn/nimbro_api/blob/main/nimbro_api/examples/tutorial.ipynb).


### Quick Start

Set the API key of one of the predefined endpoints that you want to use (`OPENAI_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_API_KEY`)

`export OPENAI_API_KEY='yourkey'`

Launch the the main launch file

`ros2 launch nimbro_api launch.py`

Instantiate an [ApiDirector](https://github.com/AIS-Bonn/nimbro_api/tree/main/nimbro_api/api_director.py) object in your Python node that exposes all package features as functions

`from nimbro_api.api_director import ApiDirector`

`self.api_director = ApiDirector(self)`

Acquire a Completions node and prompt it

`success, message, completions_id = self.api_director.acquire(reset_parameters=True, reset_context=True)`

`success, message, text_response, tool_calls = self.api_director.prompt(completions_id=completions_id, text='Tell me a joke about PhD students.', response_type="text")`

Or get some text embeddings

`success, message, embeddings = self.api_director.get_embeddings(text=["dog", "helicopter"])`

## TODOs

Here is a list of features that I would like to implement at some point:

- [ ] Vision output for Completions
- [ ] Audio input/output for Completions
- [ ] Support reasoning models (set effort, parse output, report usage)
- [ ] Structured outputs for Completions
- [ ] Random seed for Completions
- [ ] Web search options for Completions
- [ ] Action client for streamed Completions
- [ ] ApiDirector documentation

## Citation

If you find this package useful in your work, please cite one of our papers (if in doubt, choose the first one):

https://arxiv.org/abs/2503.16538
```bibtex
@article{paetzold25detector,
    author={Bastian P{\"a}tzold and Jan Nogga and Sven Behnke},
    title={Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking},
    journal={arXiv preprint arXiv:2503.16538},
    year={2025}
}
```

https://arxiv.org/abs/2410.22997
```bibtex
@article{bode24prompt,
    author={Jonas Bode and Bastian P{\"a}tzold and Raphael Memmesheimer and Sven Behnke},
    title={A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics},
    journal={International Conference on Humanoid Robots (Humanoids)},
    year={2024}
}
```

https://arxiv.org/abs/2412.14989
```bibtex
@article{memmesheimer25robocup,
    author={Raphael Memmesheimer and Jan Nogga and Bastian P{\"a}tzold and Evgenii Kruzhkov and Simon Bultmann and Michael Schreiber and Jonas Bode and Bertan Karacora and Juhui Park and Alena Savinykh and Sven Behnke},
    title={{RoboCup@Home 2024 OPL Winner NimbRo}: Anthropomorphic Service Robots using Foundation Models for Perception and Planning},
    journal={RoboCup 2024: RoboCup World Cup XXVII},
    year={2025}
}
```

## License

`nimbro_api` is licensed under BSD-3.

## Author

Bastian Pätzold <paetzold@ais.uni-bonn.de>