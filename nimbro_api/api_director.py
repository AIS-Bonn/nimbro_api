#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_prefix
from nimbro_utils.lazy import update_dict
from nimbro_api.api_director_base import ApiDirectorBase

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int).
    'severity': 20,
    # Logger suffix (str).
    'suffix': "api_director",
    # Time in seconds to wait for services to become available and for simple services to respond (float | int).
    'timeout_service': 5,
    # Time in seconds to wait for complex services to respond (float | int).
    'timeout_response': 500,
    # Name of the CompletionsMultiplexer node to use (str).
    'node_completions_multiplexer': "/nimbro_api/completions_multiplexer",
    # Name of the Embeddings node to use (str).
    'node_embeddings': "/nimbro_api/embeddings",
    # Name of the Images node to use (str).
    'node_images': "/nimbro_api/images",
    # Name of the Speech node to use (str).
    'node_speech': "/nimbro_api/speech",
    # Name of the NimbroVision node to use (str).
    'node_nimbro_vision': "/nimbro_api/nimbro_vision",
    # Name of the UsageMonitor node to use (str).
    'node_usage_monitor': "/nimbro_api/usage_monitor",
    # Path to a voice presets file (str) or a dict of the same format (dict).
    'voice_presets': os.path.join(get_package_prefix("nimbro_api").replace("install", "src"), "nimbro_api", "misc", "voice_presets.json")
}

class ApiDirector(ApiDirectorBase):
    """
    ROS2 client interface for accessing various AI model APIs through service calls.

    Provides a robust and flexible Python interface to the Chat Completions API,
    Embeddings API, Images API, Speech API, and NimbRo Vision API, as well as
    usage monitoring capabilities and support for asynchronous operations,
    abstracting away ROS2 service communication details.
    """

    def __init__(self, node, settings=None):
        """
        Initialize the ApiDirector with a ROS2 node and optional settings.

        Args:
            node (rclpy.node.Node): The ROS2 node to use for service communication.
            settings (dict | None, optional): Configuration settings to override defaults.
            Will be merged with `default_settings`. Defaults to None.

        Raises:
            AssertionError: If arguments are invalid.
        """
        settings = update_dict(old_dict=default_settings, new_dict=settings)
        super().__init__(node=node, settings=settings)

    # ApiDirector Settings

    def get_settings(self):
        """
        Retrieve the current settings of the ApiDirector.

        Returns:
            dict: A deep copy of the current settings.
        """
        return self._get_settings()

    def set_settings(self, settings, keep_existing=True):
        """
        Update the settings of the ApiDirector.

        Args:
            settings (dict): New settings to apply.
            keep_existing (bool, optional): If True, merge with existing settings.
                Otherwise, replace current settings entirely. Defaults to True.

        Raises:
            AssertionError: If input arguments or provided settings are invalid.
        """
        return self._set_settings(settings, keep_existing)

    # Chat Completions API - Management

    def get_status(self, retry=False):
        """
        Get the status of all completions nodes.

        Args:
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, list[str] | None, list[bool] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - completions_ids (list[str] | None): List of IDs of all multiplexed
                  completions nodes, or None if failed.
                - acquired (list[bool] | None): Corresponding list indicating if each
                  completions node is currently acquired, or None if failed.
        """
        return self._get_status(retry)

    def acquire(self, reset_parameters=True, reset_context=True, retry=False):
        """
        Acquire a new completions node.

        Args:
            reset_parameters (bool, optional): Whether to reset all parameters
                of the acquired node to their initial values. Defaults to True.
            reset_context (bool, optional): Whether to clear the context of
                the acquired node. Defaults to True.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - completions_id (str | None): The ID of the newly acquired completions
                  node, or None if failed.
        """
        return self._acquire(reset_parameters, reset_context, retry)

    def duplicate(self, completions_id, retry=False):
        """
        Create a duplicate of an existing completions node with the same configuration (parameters, tools, context).

        Args:
            completions_id (str): The unique identifier of the completions node to duplicate.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - new_completions_id (str | None): The ID of the newly created duplicate
                  completions node, or None if failed.
        """
        return self._duplicate(completions_id, retry)

    def release(self, completions_id=None, retry=False):
        """
        Release one or all acquired completions nodes.

        Args:
            completions_id (str | None, optional): The unique identifier of the completions
                node to release. If None, releases all acquired nodes. Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
        """
        return self._release(completions_id, retry)

    # Chat Completions API - Prompting

    def prompt(self, completions_id, text, role="user", reset_context=False, tool_response_id=None, response_type="auto", identifier=None, retry=False):
        """
        Prompt a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node to use.
            text (str | dict | list): The prompt content. Must be a string when `role` is
                'system', 'user', 'assistant', or 'tool'. Must be a dict (single message) or list
                of dicts (multiple messages) when `role` is 'json', following the API documentation.
            role (str, optional): The role of the added message. Must be one of 'system',
                'user', 'assistant', 'tool', or 'json'. Defaults to 'user'.
            reset_context (bool, optional): Whether to clear the context of the completions
                node before adding the provided message/s. Defaults to False.
            tool_response_id (str | None, optional): The ID of a previous tool call that the
                provided message is responding to. Required when `role` is 'tool'. Defaults to None.
            response_type (str | None, optional): Controls the expected response format.
                Must be 'none' (no response, just add to context), 'text' (response must be
                text), 'tool name' (response must call specified tool), 'auto' (whatever),
                'always' (response must contain at least one tool call), or None. Defaults to 'auto'.
            identifier (str | None, optional): An identifier passed to the usage monitor
                for tracking usage. Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - completion (dict | None): The parsed response containing 'text_response'
                  (str) or 'tool_response' (list) if successful, or None if failed.
        """
        return self._prompt(completions_id, text, role, reset_context, tool_response_id, response_type, identifier, retry)

    def async_prompt(self, completions_id, text, role="user", reset_context=False, tool_response_id=None, response_type="auto", identifier=None, retry=False, succeed_async_id=None):
        """
        Prompt a completions node asynchronously.

        Args:
            completions_id (str): The unique identifier of the completions node to use.
            text (str | dict | list): The prompt content. Must be a string when `role` is
                'system', 'user', 'assistant', or 'tool'. Must be a dict (single message) or
                list of dicts (multiple messages) when `role` is 'json'.
            role (str, optional): The role of the added message. Must be one of 'system',
                'user', 'assistant', 'tool', or 'json'. Defaults to 'user'.
            reset_context (bool, optional): Whether to clear the context of the completions
                node before adding the provided message/s. Defaults to False.
            tool_response_id (str | None, optional): The ID of a previous tool call that the
                provided message is responding to. Required when `role` is 'tool'. Defaults to None.
            response_type (str | None, optional): Controls the expected response format.
                Must be 'none' (no response, just add to context), 'text' (response must be
                text), 'tool name' (response must call specified tool), 'auto' (whatever),
                'always' (response must contain at least one tool call), or None. Defaults to 'auto'.
            identifier (str | None, optional): An identifier passed to the usage monitor
                for tracking usage. Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.
            succeed_async_id (str | None, optional): ID of another async operation that
                must complete successfully before this one starts. Defaults to None.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the async operation was registered, False otherwise.
                - message (str): A descriptive message about the operation result.
                - async_id (str | None): The ID for retrieving the async result later,
                  or None if failed.
        """
        return self._async_prompt(completions_id, text, role, reset_context, tool_response_id, response_type, identifier, retry, succeed_async_id)

    def interrupt(self, completions_id, retry=False):
        """
        Interrupt an ongoing generation of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node to interrupt.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
        """
        return self._interrupt(completions_id, retry)

    # Chat Completions API - Tools

    def get_tools(self, completions_id, retry=False):
        """
        Get all tool definitions of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, list[dict] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - tools (list[dict] | None): List of all tool definitions for the
                  completions node, or None if failed.
        """
        return self._get_tools(completions_id, retry)

    def set_tools(self, completions_id, tools, retry=False):
        """
        Set the tool definitions of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node.
            tools (list[dict] | None): The tool definitions to set. Must follow the
                required format as specified in the API documentation.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
        """
        return self._set_tools(completions_id, tools, retry)

    def async_set_tools(self, completions_id, tools, retry=False, succeed_async_id=None):
        """
        Set the tool definitions of a completions node asynchronously.

        Args:
            completions_id (str): The unique identifier of the completions node.
            tools (list[dict] | None): The tool definitions to set. Must follow the
                required format as specified in the API documentation.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.
            succeed_async_id (str | None, optional): ID of another async operation that
                must complete successfully before this one starts. Defaults to None.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the async operation was registered, False otherwise.
                - message (str): A descriptive message about the operation result.
                - async_id (str | None): The ID for retrieving the async result later,
                  or None if failed.
        """
        return self._async_set_tools(completions_id, tools, retry, succeed_async_id)

    # Chat Completions API - Parameters

    def get_parameters(self, completions_id, retry=False):
        """
        Get all parameters of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - parameters (dict | None): Dictionary mapping parameter names to their
                  current values, or None if failed.
        """
        return self._get_parameters(completions_id, retry)

    def reset_parameters(self, completions_id, retry=False):
        """
        Reset all parameters of a completions node to their initial values.

        Args:
            completions_id (str): The unique identifier of the completions node.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
        """
        return self._reset_parameters(completions_id, retry)

    def set_parameters(self, completions_id, parameter_names=None, parameter_values=None, retry=False):
        """
        Set parameters of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node.
            parameter_names (str | list[str] | None, optional): The names of parameters
                to set. Can be a single string or list of strings. Defaults to None.
            parameter_values (str | list[str] | None, optional): The values corresponding
                to `parameter_names`. Must be strings as correct types are inferred.
                Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
        """
        return self._set_parameters(completions_id, parameter_names, parameter_values, retry)

    def async_set_parameters(self, completions_id, parameter_names=None, parameter_values=None, retry=False, succeed_async_id=None):
        """
        Set parameters of a completions node asynchronously.

        Args:
            completions_id (str): The unique identifier of the completions node.
            parameter_names (str | list[str] | None, optional): The names of parameters
                to set. Can be a single string or list of strings. Defaults to None.
            parameter_values (str | list[str] | None, optional): The values corresponding
                to `parameter_names`. Must be strings as correct types are inferred.
                Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.
            succeed_async_id (str | None, optional): ID of another async operation that
                must complete successfully before this one starts. Defaults to None.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the async operation was registered, False otherwise.
                - message (str): A descriptive message about the operation result.
                - async_id (str | None): The ID for retrieving the async result later,
                  or None if failed.
        """
        return self._async_set_parameters(completions_id, parameter_names, parameter_values, retry, succeed_async_id)

    # Chat Completions API - Context

    def get_context(self, completions_id, retry=False):
        """
        Get the context of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, list[dict] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - context (list[dict] | None): The full context as a list of messages
                  from first to last, or None if failed.
        """
        return self._get_context(completions_id, retry)

    def set_context(self, completions_id, mode="reset", new_messages=None, index=0, indexing_last_to_first=True, retry=False):
        """
        Modify the context of a completions node.

        Args:
            completions_id (str): The unique identifier of the completions node.
            mode (str, optional): The context modification mode. Must be one of 'reset'
                (clear and replace with `new_messages`), 'insert' (insert `new_messages`
                at `index`), 'replace' (replace messages starting at `index`), or 'remove'
                (remove message at `index`). Defaults to 'reset'.
            new_messages (list[dict] | None, optional): The messages to add or replace.
                Ignored when `mode` is 'remove'. Defaults to None.
            index (int, optional): The position for insert/replace/remove operations.
                Ignored when `mode` is 'reset'. Defaults to 0.
            indexing_last_to_first (bool, optional): Whether index 0 points to the last
                (newest) or first (oldest) message in context. Defaults to True.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
        """
        return self._set_context(completions_id, mode, new_messages, index, indexing_last_to_first, retry)

    # Embeddings API

    def get_embeddings(self, text, identifier=None, retry=False):
        """
        Generate text embeddings using the Embeddings API.

        Args:
            text (str | list[str]): The text or list of texts for which embeddings
                are to be retrieved.
            identifier (str | None, optional): An identifier passed to the usage monitor
                for tracking usage. Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, list[float] | list[list[float]] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - embeddings (list[float] | list[list[float]] | None): The retrieved
                  embeddings as a single list (if input was str) or list of lists
                  (if input was list[str]), or None if failed.
        """
        return self._get_embeddings(text, identifier, retry)

    # Images API

    def get_images(self, prompt, model=None, quality=None, style=None, size=None, retry=False):
        """
        Generate an image from a text prompt using the Images API.

        Args:
            prompt (str): The prompt for which an image is to be generated.
            model (str | None, optional): The name of the model to use. Defaults to None.
            quality (str | None, optional): Quality setting supported by the model.
                Ignored if not available. Defaults to None.
            style (str | None, optional): Style setting supported by the model.
                Ignored if not available. Defaults to None.
            size (str | None, optional): Size setting supported by the model.
                Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - path (str | None): The path to the generated image file, or None if failed.
        """
        return self._get_images(prompt, model, quality, style, size, retry)

    # Speech API

    def get_speech(self, text, model=None, voice=None, speed=1.0, instructions=None, retry=False):
        """
        Generate speech from text using the Speech API.

        Args:
            text (str): The text to convert to speech.
            model (str | None, optional): The name of the model to use. Defaults to None.
            voice (str | None, optional): Voice setting supported by the model.
                Defaults to None.
            speed (float | int, optional): The speed of the speech between 0.25 and 4.0.
                Defaults to 1.0.
            instructions (str | None, optional): Instructions to control the generated
                voice. Pass a name from `voice_presets` to insert instructions accordingly.
                If `voice` is an empty string or None the preset voice is adopted as well.
                Defaults to None.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - path (str | None): The path to the generated speech file, or None if failed.
        """
        return self._get_speech(text, model, voice, speed, instructions, retry)

    # NimbRo Vision API

    def mmgroundingdino(self, image, prompts, model_id=0, model_flavor="large", min_confidence=0.0, nms_iou=0.6, overdetect_factor=1.0, retry=False):
        """
        Perform object detection using the MMGroundingDINO model served by the NimbRo Vision API.

        Args:
            image (str | list[str]): The image path or list of image paths to process.
            prompts (list[str] | list[list[str]]): The detection prompts. Can be a list
                of strings for single image or list of lists for batch processing.
            model_id (int, optional): The index of the model to use. Defaults to 0.
            model_flavor (str, optional): The flavor of the model to use. Defaults to 'large'.
            min_confidence (float | list[float], optional): Minimum confidence threshold
                for detections. Can be a single value or list for batch processing.
                Defaults to 0.0.
            nms_iou (float | list[float], optional): Non-maximum suppression IoU threshold.
                Can be a single value or list for batch processing. Defaults to 0.6.
            overdetect_factor (float | None | list[float | None], optional): Factor to control
                over-detect-factor. Can be a single value or list for batch processing.
                - Use None to not apply.
                - Use value greater zero as multiplier with number of prompts
                    for padding until desired number of detection is reached.
                - Use zero or less to only forward the best detection per prompt.
                Defaults to 1.0.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | list[dict] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - result (dict | list[dict] | None): Detection results as a single dict
                  (if single image) or list of dicts (if batch), or None if failed.
        """
        return self._mmgroundingdino(image, prompts, model_id, model_flavor, min_confidence, nms_iou, overdetect_factor, retry)

    def sam2_realtime_update(self, image, prompts, model_id=0, model_flavor="large", retry=False):
        """
        Initialize or update the SAM2 real-time model served by the NimbRo Vision API.

        Args:
            image (str): The image path to process.
            prompts (list[dict]): The tracking prompts to update the model with.
                - For box prompts use:
                    {'object_id': int, 'bbox': [x0,y0,x1,y1]}
                - For point prompts use:
                    {'object_id': int, 'points': [[x0,y0], [x1,x1]], 'labels': [1, 0]}
            model_id (int, optional): The index of the model to use. Defaults to 0.
            model_flavor (str, optional): The flavor of the model to use. Defaults to 'large'.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - result (dict | None): Tracking update results, or None if failed.
        """
        return self._sam2_realtime_update(image, prompts, model_id, model_flavor, retry)

    def sam2_realtime_track(self, image, model_id=0, retry=False):
        """
        Continue tracking with the SAM2 real-time model served by the NimbRo Vision API.

        Args:
            image (str | list[str]): The image path or list of image paths to track.
            model_id (int, optional): The index of the model to use. Defaults to 0.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | list[dict] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - result (dict | list[dict] | None): Tracking results as a single dict
                  (if single image) or list of dicts (if batch), or None if failed.
        """
        return self._sam2_realtime_track(image, model_id, retry)

    def dam(self, image, prompts, query="Describe the masked region in detail.", model_id=0, model_flavor="3B", temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=512, max_batch_size=32, retry=False):
        """
        Generate descriptions using the DAM model served by the NimbRo Vision API.

        Args:
            image (str | list[str]): The image path or list of image paths to process.
            prompts (dict | list[dict] | list[list[dict]]): The mask prompts for regions
                to describe. Can be a single dict, list of dicts, or list of lists for batch:
                    {'mask': <b64-encoded mask with shape of bbox>, 'bbox': [x0,y0,x1,y1]}
            query (str | list[str], optional): The description query. Can be a single
                string or list for batch processing. Defaults to 'Describe the masked region in detail.'.
            model_id (int, optional): The index of the model to use. Defaults to 0.
            model_flavor (str, optional): The flavor of the model to use. Defaults to '3B'.
            temperature (float | list[float], optional): Sampling temperature. Can be
                a single value or list for batch processing. Defaults to 0.2.
            top_p (float | list[float], optional): Top-p sampling parameter. Can be
                a single value or list for batch processing. Defaults to 0.5.
            num_beams (int | list[int], optional): Number of beams for beam search.
                Can be a single value or list for batch processing. Defaults to 1.
            max_new_tokens (int | list[int], optional): Maximum number of new tokens
                to generate. Can be a single value or list for batch processing. Defaults to 512.
            max_batch_size (int | list[int], optional): Maximum batch size for processing.
                Can be a single value or list for batch processing. Defaults to 32.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, str | list[str] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - result (str | list[str] | None): Generated descriptions as a single
                  string (if single image) or list of strings (if batch), or None if failed.
        """
        return self._dam(image, prompts, query, model_id, model_flavor, temperature, top_p, num_beams, max_new_tokens, max_batch_size, retry)

    def kosmos2(self, image, prompt="<grounding> Describe this image in detail:", model_id=0, model_flavor="patch14-224", num_beams=3, max_new_tokens=1024, max_batch_size=6, retry=False):
        """
        Perform vision-language tasks using the Kosmos-2 model served by the NimbRo Vision API.

        Args:
            image (str | list[str]): The image path or list of image paths to process.
            prompt (str | list[str]): The task prompt. Can be a single string or
                list of string for batch processing. Defaults to '<grounding> Describe this image in detail:'.
            model_id (int, optional): The index of the model to use. Defaults to 0.
            model_flavor (str, optional): The flavor of the model to use. Defaults to 'patch14-224'.
            num_beams (int | list[int], optional): Number of beams for beam search.
                Can be a single value or list for batch processing. Defaults to 3.
            max_new_tokens (int | list[int], optional): Maximum number of new tokens
                to generate. Can be a single value or list for batch processing. Defaults to 1024.
            max_batch_size (int | list[int], optional): Maximum batch size for processing.
                Can be a single value or list for batch processing. Defaults to 6.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | list[dict] | None, str | list[str] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - detections (dict | list[dict] | None): Detection results as a single
                  dict (if single image) or list of dicts (if batch), or None if failed.
                - captions (str | list[str] | None): Generated captions as a single
                  string (if single image) or list of strings (if batch), or None if failed.
        """
        return self._kosmos2(image, prompt, model_id, model_flavor, num_beams, max_new_tokens, max_batch_size, retry)

    def florence2(self, image, prompt, model_id=0, model_flavor="large", num_beams=3, max_new_tokens=1024, max_batch_size=6, retry=False):
        """
        Perform vision-language tasks using the Florence-2 model served by the NimbRo Vision API.

        Args:
            image (str | list[str]): The image path or list of image paths to process.
            prompt (dict | list[dict]): The task prompt. Can be a single dict or
                list of dicts for batch processing:
                - {'task_prompt': "<OD>", 'prompt_args': None}
                - {'task_prompt': "<DENSE_REGION_CAPTION>", 'prompt_args': None}
                For more info see 'https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main'.
            model_id (int, optional): The index of the model to use. Defaults to 0.
            model_flavor (str, optional): The flavor of the model to use. Defaults to 'large'.
            num_beams (int | list[int], optional): Number of beams for beam search.
                Can be a single value or list for batch processing. Defaults to 3.
            max_new_tokens (int | list[int], optional): Maximum number of new tokens
                to generate. Can be a single value or list for batch processing. Defaults to 1024.
            max_batch_size (int | list[int], optional): Maximum batch size for processing.
                Can be a single value or list for batch processing. Defaults to 6.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | list[dict] | None, str | list[str] | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - detections (dict | list[dict] | None): Detection results as a single
                  dict (if single image) or list of dicts (if batch), or None if failed.
                - captions (str | list[str] | None): Generated captions as a single
                  string (if single image) or list of strings (if batch), or None if failed.
        """
        return self._florence2(image, prompt, model_id, model_flavor, num_beams, max_new_tokens, max_batch_size, retry)

    # General

    def get_usage(self, api_type=None, api_endpoint=None, model_name=None, identifier=None, stamp_start=None, stamp_end=None, retry=False):
        """
        Retrieve usage statistics logged by the usage monitor node.

        Args:
            api_type (str | None, optional): Filter by API type. Defaults to None to
                retrieve all API types.
            api_endpoint (str | None, optional): Filter by endpoint. Defaults to None
                to retrieve all endpoints.
            model_name (str | None, optional): Filter by model name. Defaults to None
                to retrieve all models.
            identifier (str | None, optional): Filter by identifier. Defaults to None
                to retrieve all identifiers.
            stamp_start (float | int | str | datetime.datetime | rclpy.time.Time | builtin_interfaces.msg.Time | None, optional):
                Discard usage beginning before this time. Defaults to None to retrieve from first usage.
            stamp_end (float | int | str | datetime.datetime | rclpy.time.Time | builtin_interfaces.msg.Time | None, optional):
                Discard usage ending after this time. Defaults to None to retrieve until last usage.
            retry (bool | int, optional): Whether to retry on failure. If True, retries
                indefinitely. If an integer, specifies the number of retry attempts.
                Defaults to False.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - success (bool): True if the operation succeeded, False otherwise.
                - message (str): A descriptive message about the operation result.
                - usage (dict | None): The requested usage data as a parsed JSON
                  dictionary, or None if failed.
        """
        return self._get_usage(api_type, api_endpoint, model_name, identifier, stamp_start, stamp_end, retry)

    def async_get(self, async_id, mute_timeout_logging=False, timeout=None):
        """
        Retrieve the result of an asynchronous operation.

        Args:
            async_id (str): The unique identifier of the asynchronous operation.
            mute_timeout_logging (bool, optional): Whether to suppress timeout logging.
                Defaults to False.
            timeout (float | int | None, optional): Maximum time to wait for the result
                in seconds. Defaults to None for no timeout.

        Raises:
            AssertionError: If arguments are invalid.

        Returns:
            tuple[bool, str, tuple | None]: A tuple containing:
                - success (bool): True if the result was retrieved, False otherwise.
                - message (str): A descriptive message about the operation result.
                - result (tuple | None): The result tuple from the async operation,
                  or None if failed or timed out.
        """
        return self._async_get(async_id, mute_timeout_logging, timeout)

    def get_async_status(self):
        """
        Log the status of all registered asynchronous operations.

        Returns:
            tuple[bool, str]: A tuple containing:
                - success (bool): Always True for this operation.
                - message (str): A descriptive message about the async status.
        """
        return self._get_async_status()
