#!/usr/bin/env python3

import os
import time
from rclpy.impl.rcutils_logger import RcutilsLogger

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    import json
    ORJSON_AVAILABLE = False

def count_duplicates(iterable, include_unique=False):
    count_dict = {}
    for item in iterable:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    return {key: value for key, value in count_dict.items() if value > 1 or include_unique}

def count_tokens(string, encoding_name):
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def read_json(file_path, logger=None):
    assert isinstance(file_path, str), f"Expected 'file_path' to be of type 'str', but it is of type '{type(file_path).__name__}'!"
    assert isinstance(logger, RcutilsLogger) or logger is None, f"Expected 'logger' to be of type 'RcutilsLogger' or 'None', but it is of type '{type(logger).__name__}'!"

    tic = time.perf_counter()
    if logger is not None:
        logger.debug(f"Reading file '{file_path}'")

    try:
        if ORJSON_AVAILABLE:
            with open(file_path, "rb") as f:
                json_object = orjson.loads(f.read())
        else:
            if logger is not None:
                logger.warn("Using slow 'json' module to read file. Install 'orjson' to speed this up!", once=True)
            with open(file_path, 'r') as f:
                json_object = json.load(f)
    except Exception as e:
        success = False
        message = f"Failed to read file '{file_path}': {e}"
        if logger is not None:
            logger.error(message)
        json_object = None
    else:
        success = True
        message = f"Reading file '{file_path}' took '{time.perf_counter() - tic:.3f}s'."
        if logger is not None:
            logger.debug(message[:-1])

    return success, message, json_object

def write_json(file_path, json_object, indent=True, logger=None):
    assert isinstance(file_path, str), f"Expected 'file_path' to be of type 'str', but it is of type '{type(file_path).__name__}'!"
    assert isinstance(indent, bool), f"Expected 'indent' to be of type 'bool', but it is of type '{type(indent).__name__}'!"
    assert isinstance(logger, RcutilsLogger) or logger is None, f"Expected 'logger' to be of type 'RcutilsLogger' or 'None', but it is of type '{type(logger).__name__}'!"

    tic = time.perf_counter()
    if logger is not None:
        logger.debug(f"Writing file '{file_path}'")

    target_folder = os.path.dirname(file_path)
    if not os.path.exists(target_folder):
        if logger is not None:
            logger.debug(f"Creating folder '{target_folder}'")
        try:
            os.makedirs(target_folder)
        except Exception as e:
            success = False
            message = f"Failed to create target folder '{target_folder}': {e}"
            if logger is not None:
                logger.error(message)
            return success, message

    try:
        if ORJSON_AVAILABLE:
            with open(file_path, "wb") as f:
                if indent:
                    f.write(orjson.dumps(json_object, option=orjson.OPT_INDENT_2))
                else:
                    f.write(orjson.dumps(json_object))
        else:
            if logger is not None:
                logger.warn("Using slow 'json' module to write file. Install 'orjson' to speed this up!", once=True)
            with open(file_path, 'w') as f:
                json.dump(json_object, f, indent=2 if indent else None)
    except Exception as e:
        success = False
        message = f"Failed to write file '{file_path}': {e}"
        if logger is not None:
            logger.error(message)
    else:
        success = True
        message = f"Writing file '{file_path}' took '{time.perf_counter() - tic:.3f}s'."
        if logger is not None:
            logger.debug(message[:-1])

    return success, message
