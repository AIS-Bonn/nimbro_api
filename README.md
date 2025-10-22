# NimbRo API

Integration of various APIs with the [ROS2 Jazzy](https://docs.ros.org/en/jazzy/index.html) distribution.

## Features

- Supported APIs: [Chat Completions](https://platform.openai.com/docs/api-reference/chat), [Embeddings](https://platform.openai.com/docs/api-reference/embeddings), [Speech](https://platform.openai.com/docs/api-reference/speech), [Images](https://platform.openai.com/docs/api-reference/images), [NimbRoVisionServers](https://github.com/AIS-Bonn/nimbro_vision_servers).
- Supported providers: [OpenAI](https://platform.openai.com/docs/api-reference/chat), [Mistral AI](https://docs.mistral.ai/api/#tag/chat), [OpenRouter](https://openrouter.ai/docs/api-reference/overview), [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai), or custom ones behaving similar.
- The integration of the [Chat Completions](https://platform.openai.com/docs/api-reference/chat) API supports: Reasoning, (parallel) tool calling, JSON mode, image/audio/file inputs, web search, streaming, model parameters, context editing, [custom parsers](./nimbro_api/misc/parsers/completion/completion_parser_template.py), error correction, robust timeout behavior, etc.
- Easy Python bindings in a central object ([ApiDirector](./nimbro_api/api_director.py)) attachable to your node.
- A [Jupyter Notebook](./examples/tutorial.ipynb) with examples and descriptions of most features provided.
- Tracking of token usage with [cost estimation](./nimbro_api/misc/pricing.json).
- Caching responses to reduce latency and costs.
- Lite [dependencies](./requirements.txt).

## Setup

### ROS2

Include this repository together with [NimbRo API Interfaces](https://github.com/AIS-Bonn/nimbro_api_interfaces) and [NimbRo Utilities](https://github.com/AIS-Bonn/nimbro_utils) in the source folder of your colcon workspace. After building them:
```bash
colcon build --packages-select nimbro_utils nimbro_api_interfaces nimbro_api --symlink-install
```
and re-sourcing:
```bash
source install/local_setup.bash
```
several [launch files](./launch) and [nodes](./examples) will be available in your environment.

### Python

The only strictly required Python dependency of this package is the `requests` package:
```bash
pip install requests
```

To install this and all other [optional](./requirements.txt) Python dependencies:
```bash
pip install -r requirements.txt
```

<!-- ### Docker

Alternatively, you may use the provided [devcontainer](./.devcontainer) or [Docker](./Docker) image:
```bash
TODO
``` -->

### Quick Start

Set the API key for the provider you want to use (`OPENAI_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_API_KEY`, `AIS_API_KEY`, `NIMBRO_VISION_API_KEY`):
```bash
export OPENAI_API_KEY='MyKey123'
```

Launch the the main launch-file:
```bash
ros2 launch nimbro_api launch.py
```

Attach an [ApiDirector](./nimbro_api/api_director.py) to your ROS2 node:
```python
from nimbro_api import ApiDirector
self.api_director = ApiDirector(self) # `self` is your Node object
```

Use it to generate embeddings:
```python
success, message, embeddings = self.api_director.get_embeddings(text=["cat", "robot"])
```

or chat with your favorite model:
```python
success, message, completions_id = self.api_director.acquire()
assert success, message

success, message = self.api_director.set_parameters(
    completions_id=completions_id,
    parameter_names=["api_endpoint", "model_name", "stream_completion"],
    parameter_values=["OpenAI", "gpt-5", "False"]
)
assert success, message

success, message, completion = self.api_director.prompt(
    completions_id=completions_id,
    text='Tell me a joke about robots!'
)
```

## TODOs

Features that I would like to see implemented:
- [ ] Action client for streamed Chat Completions
- [ ] Context parsers for Chat Completions
- [ ] Support Transcriptions API
- [ ] Audio/Vision output for Chat Completions
- [ ] Structured outputs beyond tools for Chat Completions
- [ ] Configurable random seed for Chat Completions

## Citation

If you utilize this package in your research, please cite one of our relevant publications.

* **Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking**<br>
    [[arXiv:2503.16538](https://arxiv.org/abs/2503.16538)]
    ```bibtex
    @article{paetzold25vlmgist,
        author={Bastian P{\"a}tzold and Jan Nogga and Sven Behnke},
        title={Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking},
        journal={IEEE Robotics and Automation Letters (RA-L)},
        volume={10},
        number={11},
        pages={11578-11585},
        year={2025}
    }
    ```

* **A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics**<br>
    [[arXiv:2410.22997](https://arxiv.org/abs/2410.22997)]
    ```bibtex
    @article{bode24prompt,
        author={Jonas Bode and Bastian P{\"a}tzold and Raphael Memmesheimer and Sven Behnke},
        title={A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics},
        journal={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
        pages={309-314},
        year={2024}
    }
    ```

* **RoboCup@Home 2024 OPL Winner NimbRo: Anthropomorphic Service Robots using Foundation Models for Perception and Planning**<br>
    [[arXiv:2412.14989](https://arxiv.org/abs/2412.14989)]
    ```bibtex
    @article{memmesheimer25robocup,
        author={Raphael Memmesheimer and Jan Nogga and Bastian P{\"a}tzold and Evgenii Kruzhkov and Simon Bultmann and Michael Schreiber and Jonas Bode and Bertan Karacora and Juhui Park and Alena Savinykh and Sven Behnke},
        title={{RoboCup@Home 2024 OPL Winner NimbRo}: Anthropomorphic Service Robots using Foundation Models for Perception and Planning},
        journal={RoboCup 2024: RoboCup World Cup XXVII},
        volume={15570},
        pages={515-527},
        year={2025}
    }
    ```

## License

`nimbro_api` is licensed under the BSD-3-Clause License.

## Author

Bastian Pätzold <paetzold@ais.uni-bonn.de>