#!/usr/bin/env python3

import re
import sys
import json

# Some VLMs are capable of object grounding by pointing, 2d/3d bounding boxes or masks.
# This completion parser extracts the grounding content from the text completion of Qwen3-VL-style models and copies it to the grounding completion.
# This way its possible to conveniently use the model as an open vocabulary detector.
# Each grounded object has one of the following forms:
# {'x': x, 'y': y, 'label': label, 'type': 'point_2d_normalized'}
# {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': label, 'type': 'bbox_2d_normalized'}
# {'center_x': cx, 'center_y': cy, 'center_z': cz, 'size_x': sx, 'size_y': sy, 'size_z': sz, 'roll': roll, 'pitch': pitch, 'yaw': yaw, 'label': label, 'type': 'bbox_3d_in_camera'}
# Any additional key found for a grounded object is extracted as well

def _iter_json_arrays(text):
    """Yield JSON arrays found in fenced ```json blocks"""
    for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.I | re.S):
        inner = m.group(1).strip()
        # yield if the fenced block itself is a JSON array
        try:
            arr = json.loads(inner)
            if isinstance(arr, list):
                yield arr
                continue
            if isinstance(arr, dict) and any(k in arr for k in ('point_2d', 'bbox_2d', 'bbox_3d')):
                yield [arr]
                continue
        except Exception:
            pass
        # attempt extraction if the fenced block contains a JSON array substring
        for n in re.finditer(r"\[\s*{.*?}\s*\]", inner, re.S):
            try:
                arr = json.loads(n.group(0))
                if isinstance(arr, list):
                    yield arr
            except Exception:
                pass

def extract_grounding(text):
    """Extract XML <points ...>...</points> and JSON point/bbox arrays. Normalize according to Qwen3-VL convention."""
    results = []

    # XML points2d
    pattern = re.compile(r'<points\s+([^>]*)>([^<]+)</points>', re.I | re.S)
    for attrs, label in pattern.findall(text):
        label = label.replace("_", " ").replace(".", "")
        xs = {int(i): float(v) / 1000. for i, v in re.findall(r'\bx(\d+)\s*=\s*"([\d.]+)"', attrs)}
        ys = {int(i): float(v) / 1000. for i, v in re.findall(r'\by(\d+)\s*=\s*"([\d.]+)"', attrs)}
        for i in sorted(xs.keys() & ys.keys()):
            results.append({
                'x': xs[i], 'y': ys[i],
                'label': label,
                'type': 'point_2d_normalized',
            })

    # JSON arrays, point2d, bbox2d or bbox3d
    for arr in _iter_json_arrays(text):
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            # point_2d
            if 'point_2d' in obj and isinstance(obj['point_2d'], (list, tuple)) and len(obj['point_2d']) >= 2:
                x, y = obj['point_2d'][:2]
                item = {
                    'x': float(x) / 1000.0,
                    'y': float(y) / 1000.0,
                    'label': obj.get('label', ''),
                    'type': 'point_2d_normalized',
                }
                # forward extra attributes
                for k, v in obj.items():
                    if k != 'point_2d':
                        item[k] = v
                results.append(item)

            # bbox_2d
            if 'bbox_2d' in obj and isinstance(obj['bbox_2d'], (list, tuple)) and len(obj['bbox_2d']) >= 4:
                x1, y1, x2, y2 = obj['bbox_2d'][:4]
                item = {
                    'x1': float(x1) / 1000.0, 'y1': float(y1) / 1000.0,
                    'x2': float(x2) / 1000.0, 'y2': float(y2) / 1000.0,
                    'label': obj.get('label', ''),
                    'type': 'bbox_2d_normalized',
                }
                for k, v in obj.items():
                    if k != 'bbox_2d':
                        item[k] = v
                results.append(item)

            # bbox_3d
            if 'bbox_3d' in obj and isinstance(obj['bbox_3d'], (list, tuple)) and len(obj['bbox_3d']) >= 9:
                cx, cy, cz, sx, sy, sz, roll, pitch, yaw = map(float, obj['bbox_3d'][:9])
                item = {
                    'center_x': cx, 'center_y': cy, 'center_z': cz,
                    'size_x': sx, 'size_y': sy, 'size_z': sz,
                    'roll': roll, 'pitch': pitch, 'yaw': yaw,
                    'label': obj.get('label', ''),
                    'type': 'bbox_3d_in_camera',
                }
                for k, v in obj.items():
                    if k != 'bbox_3d':
                        item[k] = v
                results.append(item)

    return results

response = json.loads(sys.stdin.read())

logs = []
if 'text' in response['completion']:
    if isinstance(response['completion']['text'], str):
        grounding_content = extract_grounding(response['completion']['text'])
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
