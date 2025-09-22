#!/usr/bin/env python3

import sys
import json

# Parse response from stdin
response = json.loads(sys.stdin.read())

# Response structure
assert isinstance(response, dict)
assert set(response.keys()) == {"success", "message", "completion"}
assert isinstance(response['success'], bool)
assert response['success'] is True
assert isinstance(response['message'], str)
assert isinstance(response['completion'], dict)
assert 'logs' in response['completion']
assert isinstance(response['completion']['logs'], list)
if 'reasoning' in response['completion']:
    assert isinstance(response['completion']['reasoning'], str)
if 'text' in response['completion']:
    assert isinstance(response['completion']['text'], (str, dict))
if 'tool_calls' in response['completion']:
    assert isinstance(response['completion']['tool_calls'], list)
    assert len(response['completion']['tool_calls']) > 0
    for tool_call in response['completion']['tool_calls']:
        assert isinstance(tool_call, dict)
        assert set(tool_call.keys()) == {"id", "name", "arguments"}
        assert isinstance(tool_call['id'], str)
        assert len(tool_call['id']) > 0
        assert isinstance(tool_call['name'], str)
        assert len(tool_call['name']) > 0
        assert isinstance(tool_call['arguments'], dict)
if 'usage' in response['completion']:
    assert isinstance(response['completion']['usage'], dict)

# Modify response without violating any of the following restrictions:
# - Value of response key 'success' must be of type 'bool'.
# - Value of response key 'message' must be of type 'str'.
# - Value of response key 'completions' must be of type 'dict'.
# - Additional keys in response are omitted.
# - Response must be serializable.
# - Script must terminate before timeout.
# - Script must terminate with code '0'.

# Pass text to be logged in the Completions Node as INFO to stderr
sys.stderr.write("This parser does nothing except logging this message in the Completions Node.")

# Return response to stdout
sys.stdout.write(json.dumps(response))
