#!/usr/bin/env python3

import os
import copy
import json
import time
import datetime

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_msgs.msg import String

from nimbro_api_interfaces.msg import Embedding
from nimbro_api_interfaces.srv import EmbeddingsGet
from nimbro_api.misc.common import validate_default_endpopints, filter_api_endpoint, validate_api_endpoint, retrieve_api_key, probe_models_api, validate_connection

from nimbro_utils.lazy import start_and_spin_node, ParameterHandler, Logger, read_json, write_json, count_tokens, convert_stamp, log_lines, get_package_path

### <Parameter Defaults>

node_name = "embeddings"
severity = 10

probe_api_connection = True
api_endpoint = "OpenAI"
model_name = "text-embedding-3-large"

cache_use = True
cache_read_once = True
cache_folder = os.path.join(get_package_path("nimbro_api"), "cache", "embeddings")
cache_file = "cache_embeddings_index.json"

monitor_usage = True

## non-params

line_length = 150
embeddings_per_file = 100
embeddings_name_template = "cache_embeddings_{file_id}.json"
max_texts_per_post = 100 # batches posts including more texts to mistral to stay below token limit

api_endpoints = {
    'OpenAI': {
        'api_flavor': "openai",
        'models_url': "https://api.openai.com/v1/models",
        'embeddings_url': "https://api.openai.com/v1/embeddings",
        'key_type': "environment",
        'key_value': "OPENAI_API_KEY"
    },
    'Mistral AI': {
        'api_flavor': "openai",
        'models_url': "https://api.mistral.ai/v1/models",
        'embeddings_url': "https://api.mistral.ai/v1/embeddings",
        'key_type': "environment",
        'key_value': "MISTRAL_API_KEY"
    },
    'OpenRouter': {
        'api_flavor': "openai",
        'models_url': "https://openrouter.ai/api/v1/models",
        'embeddings_url': "https://openrouter.ai/api/v1/embeddings",
        'key_type': "environment",
        'key_value': "OPENROUTER_API_KEY"
    },
    'vLLM': {
        'api_flavor': "openai",
        'models_url': "http://localhost:8000/v1/models",
        'embeddings_url': "http://localhost:8000/v1/embeddings",
        'key_type': "environment",
        'key_value': "VLLM_API_KEY"
    },
    'AIS': {
        'api_flavor': "openai",
        'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
        'embeddings_url': "https://api-code.ais.uni-bonn.de/v1/embeddings",
        'key_type': "environment",
        'key_value': "AIS_API_KEY"
    }
}

### </Parameter Defaults>

class Embeddings(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # initialize endpoints

        self.endpoint_keys_required = {'name', 'api_flavor', 'embeddings_url', 'key_type', 'key_value'}
        self.endpoint_keys_optional = {'models_url'}
        self.endpoint_key_type_values = ["environment", "plain"]
        self.endpoint_api_flavor_values = ["openai"]
        validate_default_endpopints.__get__(self)(api_endpoints)

        self.filter_api_endpoint = filter_api_endpoint.__get__(self)
        self.validate_api_endpoint = validate_api_endpoint.__get__(self)
        self.retrieve_api_key = retrieve_api_key.__get__(self)
        self.probe_models_api = probe_models_api.__get__(self)
        self.validate_connection = validate_connection.__get__(self)

        self.api_endpoints = api_endpoints
        self.endpoint_probes = {}

        # declare parameters

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
            name="probe_api_connection",
            dtype=bool,
            default_value=probe_api_connection,
            description="Probes the Models API of the endpoint to validate the API key and model name.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="api_endpoint",
            dtype=str,
            default_value=api_endpoint,
            description=f"Sets the API endpoint defining API flavor, Models & Embeddings URLs, key type and value. Must be a valid JSON encoded dictionary or a name in {list(api_endpoints.keys())}.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="model_name",
            dtype=str,
            default_value=model_name,
            description="Name of the model that is used.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_use",
            dtype=bool,
            default_value=cache_use,
            description="Attempt to retrieve embeddings from cached results.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="cache_read_once",
            dtype=bool,
            default_value=cache_read_once,
            description="Read embeddings cache file once when required and keep it in memory instead of loading it every time.",
            read_only=False
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
            name="monitor_usage",
            dtype=bool,
            default_value=monitor_usage,
            description="Tokenize input strings to monitor usage.",
            read_only=False
        )

        # create interfaces

        qos_profile_srv = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=7)
        self.srv_embeddings = self.create_service(EmbeddingsGet, f"{self.node_namespace}/{self.node_name}/get_embeddings".replace("//", "/"), self.get_embeddings_callack, qos_profile=qos_profile_srv, callback_group=ReentrantCallbackGroup())

        qos_profile_pub = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_ALL, depth=10)
        self.pub_usage = self.create_publisher(String, f"{self.node_namespace}/api_usage".replace("//", "/"), qos_profile=qos_profile_pub, callback_group=MutuallyExclusiveCallbackGroup())

        self._logger.info("Node started")

    def __del__(self):
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self._logger.set_settings(settings={'severity': value})

        elif name == "probe_api_connection":
            if is_declared and value != self.parameters.probe_api_connection:
                self._logger.debug("Reset endpoint probes")
                self.endpoint_probes = {}

        elif name == "api_endpoint":
            value, message = self.filter_api_endpoint(name, value, line_length)

        elif name == "cache_use":
            self.index = None

        elif name == "cache_read_once":
            if value is False:
                self.index = None

        elif name == "cache_folder":
            if value == "":
                value = os.path.join(get_package_path("nimbro_api"), "cache")

        return value, message

    # Embeddings Pipeline

    def embeddings_post(self, texts, model, api_url, api_key):
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }

        data = {
            'input': texts,
            'model': model,
            'encoding_format': "float"
        }

        self._logger.debug(f"Posting request: {data}\n to '{api_url}'")
        tic = time.perf_counter()
        try:
            response = requests.post(api_url, headers=headers, json=data, stream=False)
        except Exception as e:
            toc = time.perf_counter()
            self._logger.debug(f"Error occurred after '{toc - tic:.3f}s': {repr(e)}")
            success = False
            message = f"Failed to POST request: {repr(e)}"
            embeddings = None
        else:
            toc = time.perf_counter()
            self._logger.debug(f"Received response after '{toc - tic:.3f}s'")

            if response.status_code == 200:
                response = response.json()
                success = True
                message = f"Retrieved '{len(response['data'])}' embedding{'' if len(response['data']) == 1 else 's'}."
                embeddings = [response['data'][i]['embedding'] for i in range(len(response['data']))]
            else:
                success = False
                message = f"HTTP-Error: {response.text}"
                embeddings = None

        return success, message, embeddings

    def save_usage(self, texts, identifier, stamp_start):
        stamp_stop = datetime.datetime.now()

        if self.parameters.monitor_usage:
            num_tokens = 0
            tic = time.perf_counter()
            for text in texts:
                try:
                    num_tokens += count_tokens(string=text, encoding_name="cl100k_base") # for third-generation embedding
                except ModuleNotFoundError:
                    self._logger.warn("Cannot monitor usage because the module tiktoken is not installed")
                    self.parameter_handler.update(name="monitor_usage", value=False)
                    return
            self._logger.debug(f"Tokenizing uncached text{'' if len(texts) == 1 else 's'} took '{time.perf_counter() - tic:.3f}s'")
        else:
            return

        usage = {}
        usage['api_type'] = "embeddings"
        usage['api_endpoint'] = self.parameters.api_endpoint
        usage['model_name'] = self.parameters.model_name
        if identifier != "":
            usage['identifier'] = identifier
        usage['stamp_start'] = convert_stamp(stamp=stamp_start, target_format="iso")
        usage['stamp_stop'] = convert_stamp(stamp=stamp_stop, target_format="iso")
        usage['duration'] = (stamp_stop - stamp_start).total_seconds()
        usage['tokens_input_uncached'] = num_tokens

        usage_str = json.dumps(usage, indent=4)
        log_lines(f"Usage:\n{usage_str}", line_length=150, line_highlight="|", block_format=False, logger=self._logger, severity=10)

        usage_msg = String()
        usage_msg.data = usage_str
        self.pub_usage.publish(usage_msg)

    def get_embeddings(self, texts, identifier):
        # parse argument

        if len(texts) == 0:
            return True, "Retrieved '0' embeddings.", []
        if not all(isinstance(t, str) for t in texts):
            message = "All items in list 'texts' must be of type 'str'."
            self._logger.error(message[:-1])
            return False, message, None
        text_formatted = [t.replace("\n", " ") for t in texts]

        for t in text_formatted:
            if t == "":
                message = "None of the passed texts must be empty."
                self._logger.error(message[:-1])
                return False, message, None

        embeddings = [None] * len(text_formatted)

        # read cache

        cache_use = copy.copy(self.parameters.cache_use)

        if not cache_use:
            self.index = None
            missing_idx = list(range(len(text_formatted)))
        else:
            if not self.parameters.cache_read_once or self.index is None:
                # read index file
                index_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)
                if os.path.exists(index_path):
                    success, _, self.index = read_json(file_path=index_path, logger=self._logger)
                    if success:
                        if not isinstance(self.index, dict):
                            success = False
                            self._logger.error(f"Expected content of index file to be of type 'dict', but it is of type '{type(self.index).__name__}'", throttle_duration_sec=10.0)
                        elif 'files' not in self.index or 'texts' not in self.index:
                            success = False
                            self._logger.error("Expected index file to feature the keys 'files' and 'texts'", throttle_duration_sec=10.0)
                        elif not isinstance(self.index['files'], list) or not isinstance(self.index['texts'], dict):
                            success = False
                            self._logger.error("Expected index file keys 'files' and 'texts' to feature values of type 'list' and 'dict'", throttle_duration_sec=10.0)
                    if not success:
                        self._logger.warn("Initializing new index file. Corrupt cache files might get overwritten!", throttle_duration_sec=10.0)
                        self.index = {'files': [], 'texts': {}}
                else:
                    success = False
                    self._logger.info("Initializing new index file")
                    self.index = {'files': [], 'texts': {}}

                self.embeddings_files = {}

            # assess embeddings listed in index

            corrupted = False

            if self.index['texts'].get(self.parameters.model_name) is None:
                self._logger.debug(f"Cannot find model '{self.parameters.model_name}' in index file")
                missing_idx = list(range(len(text_formatted)))
            else:
                missing_idx = []
                missing_index_tuples = []
                missing_file_idx = []
                for i, t in enumerate(text_formatted):
                    if self.index['texts'][self.parameters.model_name].get(t) is None:
                        self._logger.debug(f"Cannot find text '{t}' in index file")
                        missing_idx.append(i)
                    else:
                        if not isinstance(self.index['texts'][self.parameters.model_name][t], list):
                            self._logger.error(f"Cannot find text '{t}' in index file: File is corrupted (value of text is not a list)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        elif not len(self.index['texts'][self.parameters.model_name][t]) == 2:
                            self._logger.error(f"Cannot find text '{t}' in index file: File is corrupted (value of text is not of length 2)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        elif not isinstance(self.index['texts'][self.parameters.model_name][t][0], int) or not isinstance(self.index['texts'][self.parameters.model_name][t][1], int):
                            self._logger.error(f"Cannot find text '{t}' in index file: File is corrupted (values of text is not a list of integers)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        elif self.index['texts'][self.parameters.model_name][t][0] >= len(self.index['files']):
                            self._logger.error(f"Cannot find text '{t}' in index file: File is corrupted (value of text points to file that does not exist)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(i)
                        else:
                            file_id = self.index['texts'][self.parameters.model_name][t][0]
                            embedding_id = self.index['texts'][self.parameters.model_name][t][1]
                            self._logger.debug(f"Text '{t}' found in index file (file: '{file_id}', element: '{embedding_id}')")
                            missing_index_tuples.append((file_id, embedding_id, i))
                            missing_file_idx.append(file_id)

                # read embeddings files

                missing_file_idx = list(set(missing_file_idx))

                for i in missing_file_idx:
                    if i not in self.embeddings_files:
                        if not isinstance(self.index['files'][i], list):
                            self._logger.error(f"Cannot find embeddings file '{i}' in index file: File is corrupted (value of file is not a list)", throttle_duration_sec=10.0)
                            corrupted = True
                        elif not len(self.index['files'][i]) == 2:
                            self._logger.error(f"Cannot find embeddings file '{i}' in index file: File is corrupted (value of file is not of length 2)", throttle_duration_sec=10.0)
                            corrupted = True
                        elif not isinstance(self.index['files'][i][0], str):
                            self._logger.error(f"Cannot find embeddings file '{i}' in index file: File is corrupted (value of first element is not a string)", throttle_duration_sec=10.0)
                            corrupted = True
                        else:
                            embeddings_file_path = os.path.join(self.parameters.cache_folder, self.index['files'][i][0])
                            success, _, embeddings_file = read_json(file_path=embeddings_file_path, logger=self._logger)
                            if not success:
                                pass
                            elif not isinstance(embeddings_file, list):
                                self._logger.warn(f"Embeddings file '{embeddings_file_path}' is corrupted (content is not a list) (1)", throttle_duration_sec=10.0)
                                corrupted = True
                            elif len(embeddings_file) != self.index['files'][i][1]:
                                self._logger.warn(f"Embeddings file '{embeddings_file_path}' is corrupted (actual size '{len(embeddings_file)}' does not match size in index '{self.index['files'][i][1]}') (1)", throttle_duration_sec=10.0)
                                corrupted = True
                            else:
                                self.embeddings_files[i] = embeddings_file

                # collect embeddings

                for file_id, embedding_id, missing_id in missing_index_tuples:
                    if file_id in self.embeddings_files:
                        if embedding_id < len(self.embeddings_files[file_id]):
                            self._logger.debug(f"Found cached embedding for text '{text_formatted[missing_id]}' (file: '{file_id}', element: '{embedding_id}')")
                            embeddings[missing_id] = self.embeddings_files[file_id][embedding_id]
                        else:
                            self._logger.error(f"Cannot find embedding '{embedding_id}' in embeddings file '{file_id}': File is corrupted (file only contains '{len(self.embeddings_files[file_id])}' embeddings)", throttle_duration_sec=10.0)
                            corrupted = True
                            missing_idx.append(missing_id)
                    else:
                        missing_idx.append(missing_id)

        if len(missing_idx) == 0:
            self._logger.debug("All requested embeddings were found in cache")
        else:
            # validate connection
            if self.parameters.probe_api_connection:
                success, message = self.validate_connection(model=self.parameters.model_name)
                if not success:
                    return False, message, None

            # retrieve API key
            success, message, api_key = self.retrieve_api_key()
            if not success:
                self._logger.error(message)
                return False, message, None

            # retrieve missing embeddings

            self._logger.info(f"Retrieving '{len(missing_idx)}' missing embedding{'' if len(missing_idx) == 1 else 's'} from API")
            missing_texts = [text_formatted[i] for i in missing_idx]

            floor_mod = (len(missing_texts) // max_texts_per_post, len(missing_texts) % max_texts_per_post)
            missing_embeddings = []
            for i in range(floor_mod[0]):
                stamp_start = datetime.datetime.now()

                texts_post = missing_texts[i * max_texts_per_post: (i + 1) * max_texts_per_post]
                self._logger.info(f"Retrieving partial batch with '{len(texts_post)}' missing embedding{'' if len(texts_post) == 1 else 's'} (got '{len(missing_embeddings)}')")
                success, message, embeddings_batch = self.embeddings_post(
                    texts=texts_post,
                    model=self.parameters.model_name,
                    api_url=self.api_endpoints[self.parameters.api_endpoint]['embeddings_url'],
                    api_key=api_key
                )
                if success:
                    missing_embeddings += embeddings_batch
                else:
                    self._logger.error(message)
                    return False, message, None
            if floor_mod[1] > 0:
                stamp_start = datetime.datetime.now()

                texts_post = missing_texts[floor_mod[0] * max_texts_per_post:]
                if floor_mod[0] > 0:
                    self._logger.info(f"Retrieving partial batch with '{len(texts_post)}' missing embedding{'' if len(texts_post) == 1 else 's'} (got '{len(missing_embeddings)}')")

                success, message, embeddings_batch = self.embeddings_post(
                    texts=texts_post,
                    model=self.parameters.model_name,
                    api_url=self.api_endpoints[self.parameters.api_endpoint]['embeddings_url'],
                    api_key=api_key
                )
                if success:
                    missing_embeddings += embeddings_batch
                else:
                    self._logger.error(message)
                    return False, message, None
            if success:
                self._logger.debug(f"Retrieved '{len(missing_idx)}' missing embedding{'' if len(missing_idx) == 1 else 's'} from API")
                self.save_usage(missing_texts, identifier, stamp_start)
            else:
                self._logger.error(message)
                return False, message, None

            # fill up missing embeddings

            for i, j in enumerate(missing_idx):
                embeddings[j] = missing_embeddings[i]

            # add missing_embeddings to cache and embeddings_files, and overwrite touched files

            if cache_use and len(missing_idx) > 0 and self.index is not None and not corrupted:
                touched_embeddings_files = []
                for i in missing_idx:
                    for j, (file_name, file_size) in enumerate(self.index['files']):
                        if file_size < embeddings_per_file:
                            if j not in self.embeddings_files and j:
                                embeddings_file_path = os.path.join(self.parameters.cache_folder, self.index['files'][j][0])
                                success, _, embeddings_file = read_json(file_path=embeddings_file_path, logger=self._logger)
                                if not success:
                                    pass
                                elif not isinstance(embeddings_file, list):
                                    self._logger.warn(f"Embeddings file '{embeddings_file_path}' is corrupted (content is not a list) (2)", throttle_duration_sec=10.0)
                                    self.parameters.cache_use = False
                                    corrupted = True
                                    break
                                elif len(embeddings_file) != self.index['files'][j][1]:
                                    self._logger.warn(f"Embeddings file '{embeddings_file_path}' is corrupted (actual size '{len(embeddings_file)}' does not match size in index '{self.index['files'][i][1]}') (2)", throttle_duration_sec=10.0)
                                    self.parameters.cache_use = False
                                    corrupted = True
                                    break
                                else:
                                    self.embeddings_files[j] = embeddings_file

                            if j in self.embeddings_files:
                                self.index['files'][j][1] += 1
                                self.embeddings_files[j].append(embeddings[i])
                                if j not in touched_embeddings_files:
                                    touched_embeddings_files.append(j)
                                if self.parameters.model_name not in self.index['texts']:
                                    self.index['texts'][self.parameters.model_name] = {}
                                embedding_id = len(self.embeddings_files[j]) - 1
                                self.index['texts'][self.parameters.model_name][text_formatted[i]] = [j, embedding_id]
                                self._logger.debug(f"Caching text '{text_formatted[i]}' (file: '{j}', element: '{embedding_id}')")
                                break

                        if corrupted:
                            break

                    else:
                        j = len(self.index['files'])
                        new_name = embeddings_name_template.format(file_id=j)
                        self._logger.debug(f"Creating new embeddings file '{new_name}'")
                        self.embeddings_files[j] = [embeddings[i]]
                        if j not in touched_embeddings_files:
                            touched_embeddings_files.append(j)
                        self.index['files'].append([new_name, 1])
                        if self.parameters.model_name not in self.index['texts']:
                            self.index['texts'][self.parameters.model_name] = {}
                        embedding_id = len(self.embeddings_files[j]) - 1
                        self.index['texts'][self.parameters.model_name][text_formatted[i]] = [j, embedding_id]
                        self._logger.debug(f"Caching text '{text_formatted[i]}' (file: '{j}', element: '{embedding_id}')")

                    if corrupted:
                        break

                if not corrupted:
                    if len(touched_embeddings_files) > 0:
                        cache_path = os.path.join(self.parameters.cache_folder, self.parameters.cache_file)
                        write_json(cache_path, self.index, indent=True, logger=self._logger)
                    for i in touched_embeddings_files:
                        embeddings_file_path = os.path.join(self.parameters.cache_folder, self.index['files'][i][0])
                        write_json(embeddings_file_path, self.embeddings_files[i], indent=False, logger=self._logger)

        # forward results

        if len(str(text_formatted)) > 100:
            self._logger.info(f"Retrieved embedding{'' if len(embeddings) == 1 else 's'} for '{len(embeddings)}' text{'' if len(text_formatted) == 1 else 's'}: {str(text_formatted)[:line_length]}...")
        else:
            self._logger.info(f"Retrieved embedding{'' if len(embeddings) == 1 else 's'} for '{len(embeddings)}' text{'' if len(text_formatted) == 1 else 's'}: {text_formatted}")

        return True, f"Retrieved '{len(embeddings)}' embedding{'' if len(embeddings) == 1 else 's'}.", embeddings

    # Callbacks

    def get_embeddings_callack(self, request, response):
        self._logger.debug(f"get_embeddings_callack(): start (identifier: '{request.identifier}')")

        response.success, response.message, embeddings = self.get_embeddings(texts=request.texts, identifier=request.identifier)
        if response.success:
            for embedding_np in embeddings:
                embedding_msg = Embedding()
                embedding_msg.embedding = embedding_np
                response.embeddings.append(embedding_msg)

        self._logger.debug("get_embeddings_callack(): end")
        return response

def main(args=None):
    start_and_spin_node(Embeddings, args=args)

if __name__ == '__main__':
    main()
