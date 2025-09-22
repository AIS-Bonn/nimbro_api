#!/usr/bin/env python3

import re
import sys
import json

# Some reasoning models exhibit reasoning within the text completion rather than in a separate reasoning channel.
# Typically, this reasoning is encapsulated by special tags, such as: <think> Reasoning... </think>.
# When parsing model outputs directly, this can be problematic, e.g. when attempting to parse it as JSON.
# This completion parser extracts the reasoning content from the text completion and moves it to the reasoning completion.
# This makes the model behave like regular reasoning models.

start_str = "<think>"
end_str = "</think>"

def split_reasoning(text: str):
    reasoning_match = re.search(start_str + r"(.*?)" + end_str, text, re.DOTALL)

    if reasoning_match:
        reasoning_content = reasoning_match.group(1).strip()
        outside_content = re.sub(start_str + r".*?" + end_str, "", text, flags=re.DOTALL).strip()
    else:
        reasoning_content = ""
        outside_content = text

    return reasoning_content, outside_content

response = json.loads(sys.stdin.read())

logs = []
if 'text' in response['completion']:
    if isinstance(response['completion']['text'], str):
        reasoning_content, outside = split_reasoning(response['completion']['text'])
        if reasoning_content:
            if 'reasoning' in response['completion']:
                response['completion']['reasoning'] += f" {reasoning_content}"
                logs.append("Extracted reasoning from text completion and appended to existing reasoning completion.")
                sys.stderr.write(logs[-1])
            else:
                response['completion']['reasoning'] = reasoning_content
                logs.append("Extracted reasoning from text completion and set as reasoning completion.")
                sys.stderr.write(logs[-1])
            if outside:
                response['completion']['text'] = outside
            else:
                del response['completion']['text']
                logs.append("Removed empty text completion.")
        else:
            logs.append("There is no reasoning content in the text completion.")
            sys.stderr.write(logs[-1])
    else:
        logs.append(f"Cannot extract reasoning from text completion of type '{type(response['completion']['text']).__name__}' instead of 'str'.")
else:
    logs.append("Cannot extract reasoning without text completion.")

response['completion']['logs'] += logs

sys.stderr.write(" ".join(logs))
sys.stdout.write(json.dumps(response))
