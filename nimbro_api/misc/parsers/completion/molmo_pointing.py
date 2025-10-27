#!/usr/bin/env python3

import re
import sys
import json

# Some VLMs are capable of object grounding by pointing, 2d/3d bounding boxes or masks.
# This completion parser extracts the grounding content from the text completion of Molmo-style models and copies it to the grounding completion.
# This way its possible to conveniently use the model as an open vocabulary detector.
# Here, each grounded object has the form {'x': x_norm, 'y': y_norm, 'label': label, 'type': 'point_2d_normalized'}

def extract_points(text):
    pattern = re.compile(
        r'<point\s+([^>]*)>([^<]+)</point>'
        r'|<points\s+([^>]*)>([^<]+)</points>',
        re.I | re.S
    )
    results = []
    for m in pattern.finditer(text):
        if m.group(1): # <point ...>label</point>
            attrs, label = m.group(1), m.group(2)
            label = label.replace("_", " ").replace(".", "")
            x = re.search(r'\bx\s*=\s*"([\d.]+)"', attrs)
            y = re.search(r'\by\s*=\s*"([\d.]+)"', attrs)
            if x and y:
                results.append({
                    'x': float(x.group(1)) / 100.,
                    'y': float(y.group(1)) / 100.,
                    'label': label,
                    'type': 'point_2d_normalized'
                })
        else: # <points ...>label</points>
            attrs, label = m.group(3), m.group(4)
            label = label.replace("_", " ").replace(".", "")
            xs = {int(i): float(v) / 100. for i, v in re.findall(r'\bx(\d+)\s*=\s*"([\d.]+)"', attrs)}
            ys = {int(i): float(v) / 100. for i, v in re.findall(r'\by(\d+)\s*=\s*"([\d.]+)"', attrs)}
            for i in sorted(set(xs) & set(ys)):
                results.append({
                    'x': xs[i],
                    'y': ys[i],
                    'label': label,
                    'type': 'point_2d_normalized'
                })
    return results

response = json.loads(sys.stdin.read())

logs = []
if 'text' in response['completion']:
    if isinstance(response['completion']['text'], str):
        grounding_content = extract_points(response['completion']['text'])
        if len(grounding_content) > 0:
            if 'grounding' in response['completion']:
                response['completion']['grounding'] += grounding_content
                logs.append("Extracted grounding from text completion and appended to existing grounding completion.")
                sys.stderr.write(logs[-1])
            else:
                response['completion']['grounding'] = grounding_content
                logs.append("Extracted grounding from text completion and set as grounding completion.")
                sys.stderr.write(logs[-1])
        else:
            logs.append("There is no grounding content in the text completion.")
            sys.stderr.write(logs[-1])
    else:
        logs.append(f"Cannot extract grounding from text completion of type '{type(response['completion']['text']).__name__}' instead of 'str'.")
else:
    logs.append("Cannot extract grounding without text completion.")

response['completion']['logs'] += logs

sys.stderr.write(" ".join(logs))
sys.stdout.write(json.dumps(response))
