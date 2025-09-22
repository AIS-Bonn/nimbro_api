#!/usr/bin/env python3

import os
import json

from nimbro_utils.utility.string import normalize, levenshtein_match
from nimbro_utils.utility.misc import read_json, write_json

def semantic_match(node, source, targets, force):
    """
    TODO turn into node extension with settings including option to base matching on embeddings
    """
    assert isinstance(source, str), f"{type(source).__name__}"
    assert isinstance(targets, list) and all(isinstance(target, str) for target in targets), f"{[type(target).__name__ for target in targets]}"
    assert isinstance(force, bool), f"{type(force).__name__}"
    targets = list(set(targets))
    assert len(targets) > 0

    # TODO examples
    # TODO reasoning

    node.get_logger().info(f"Matching from '{source}' to {targets} (force={force})")

    # levensthein match
    invalid_token = "NULL"
    method = "levenshtein"
    match = levenshtein_match(
        word=source,
        labels=targets if force else targets + [invalid_token],
        threshold=0,
        normalization=True
    )
    if match is None:
        # semantic match
        method = "semantic"
        # cache semantic matches
        cache_path = os.path.join(node.data_path, "semantic_matches.json")
        if cache_path is not None:
            # read cache file
            if os.path.isfile(cache_path):
                success, _, cache = read_json(
                    file_path=cache_path,
                    logger=node.get_logger()
                )
            else:
                success = False
                node.get_logger().warn(f"Cache file for semantic matching '{cache_path}' does not exist")
            if not success:
                cache = {}

            # hash inputs
            input_dict = {
                'source': normalize(source, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, remove_white_spaces=True, lowercase=True),
                'targets': [normalize(target, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, remove_white_spaces=True, lowercase=True) for target in targets]
            }
            if not force:
                input_dict['targets'].append(invalid_token)
            input_dict = json.dumps(input_dict, sort_keys=True)
            input_hash = str(hash(input_dict))

            # return cached result
            if input_hash in cache:
                node.get_logger().info(f"Found matching result in cache: {cache[input_hash]}")
                return cache[input_hash]

        # acquire
        success, message, completions_id = node.api_director.acquire(
            reset_parameters=False,
            reset_context=False,
            retry=True
        )
        if not success:
            raise Exception(f"Unexpected failure in api_director.acquire(retry=True): {message}")
        else:
            node.get_logger().debug(f"Using completions node '{completions_id}' for matching")

        # parameters
        params = {
            'logger_level': "20",
            'probe_api_connection': "False",
            'api_endpoint': "OpenAI",
            'model_name': "gpt-4o",
            'model_temperatur': "0.0",
            'model_top_p': "1.0",
            'model_max_tokens': "200",
            'model_presence_penalty': "0.0",
            'model_frequency_penalty': "0.0",
            'stream_completion': "True",
            'normalize_text_response': "False",
            'max_tool_calls_per_response': "1",
            'correction_attempts': "2",
            'timeout_chunk': "5.0",
            'timeout_completion': "15.0"
        }
        success, message = node.api_director.set_parameters(
            completions_id=completions_id,
            parameter_names=list(params.keys()),
            parameter_values=list(params.values()),
            retry=True
        )
        if not success:
            raise Exception(f"Unexpected failure in api_director.set_parameter(retry=True): {message}")

        while True:
            # system
            prompt = "You are helpful assistant. Be concise and factual."
            success, message, _, _ = node.api_director.prompt(
                completions_id=completions_id,
                text=prompt,
                role="system",
                reset_context=True,
                tool_response_id=None,
                response_type="none",
                retry=True
            )
            if not success:
                raise Exception(f"Unexpected failure in api_director.prompt(retry=True): {message}")

            # prompt
            targets_str = ""
            for i, target in enumerate(targets):
                targets_str += f"\n{i}: {target}" # TODO (type=...)
            prompt = (
                f"I got this string: '{source}'\n\n"
                f"Please help me to determine which of the following strings is the most semantically similar to it:{targets_str}\n\n"
                "Your response must contain only the index at the beginning of the string that is the closest match from this list."
            )
            if not force:
                prompt += f"\nIf there is no obvious best match — for example, if there are no semantic similarities at all, or if there are multiple equally similar candidates — respond with '{invalid_token}'."
            success, message, response, _ = node.api_director.prompt(
                completions_id=completions_id,
                text=prompt,
                role="user",
                reset_context=False,
                tool_response_id=None,
                response_type="text",
                retry=True
            )
            if not success:
                raise Exception(f"Unexpected failure in api_director.prompt(retry=True): {message}")

            # interpret response
            match = levenshtein_match(
                word=response,
                labels=[str(i) for i in range(len(targets))] if force else [str(i) for i in range(len(targets))] + [invalid_token],
                threshold=0,
                normalization=True
            )
            if match is None:
                node.get_logger().warn(f"The response '{response}' could not be matched to any of the targets, trying again")
            else:
                if match != invalid_token:
                    match = targets[int(match)]
                break

        # release
        success, message = node.api_director.release(
            completions_id=completions_id,
            retry=True
        )
        if not success:
            raise Exception(f"Unexpected failure in api_director.release(retry=True): {message}")

    # obtain result
    if (not force) and match == invalid_token:
        result = None
    else:
        result = match
    node.get_logger().info(f"Matching result: {result} ({method})")

    # cache result
    if method == "semantic" and cache_path is not None:
        cache[input_hash] = result
        write_json(
            file_path=cache_path,
            json_object=cache,
            indent=False,
            logger=node.get_logger()
        )

    return result
