#!/usr/bin/env python3

import sys
import json

from nimbro_utils.lazy import extract_json

# This completion parser attempts to decode text completions as JSON objects.
# It provides a pseudo-JSON mode that does not affect model inference, yet still ensures that a successful response is valid JSON.

response = json.loads(sys.stdin.read())

if 'text' in response['completion']:
    if isinstance(response['completion']['text'], str):
        json_obj = extract_json(response['completion']['text'], first_over_longest=False)
        if json_obj:
            response['completion']['text'] = json_obj
            response['completion']['logs'].append("Successfully decoded text completion as JSON.")
            sys.stderr.write(response['completion']['logs'][-1])
        else:
            response['completion']['success'] = False
            response['completion']['message'] = "Failed to decode text completion as JSON."
    else:
        response['completion']['success'] = False
        response['completion']['message'] = f"Cannot decode JSON from text completion of type '{type(response['completion']['text']).__name__}' instead of 'str'."
else:
    response['completion']['success'] = False
    response['completion']['message'] = "Cannot decode JSON without text completion."

sys.stdout.write(json.dumps(response))
