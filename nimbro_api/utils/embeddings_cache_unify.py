#!/usr/bin/env python3

import os
import time

try:
    import orjson
    ORJSON_AVAILABLE = True
    print("Using fast 'orjson' module")
except ImportError:
    import json
    ORJSON_AVAILABLE = False
    print("Using slow 'json' module")

### <Settings>

# index file to read
index_path = "C:\\Users\\Me\\Desktop\\embeddings\\cache_embeddings_index.json"
index_path = "./../../cache/embeddings/cache_embeddings_index.json"

# cache file to write
cache_path = "C:\\Users\\Me\\Desktop\\cache_embeddings.json"
cache_path = "./../../cache/cache_embeddings.json"

### </Settings>

print(f"Reading index file '{index_path}'...")
tic = time.perf_counter()
try:
    if ORJSON_AVAILABLE:
        with open(index_path, "rb") as f:
            index = orjson.loads(f.read())
    else:
        with open(index_path, 'r') as f:
            index = json.load(f)
except Exception as e:
    print(f"Failed to read index '{index_path}': {e}")
    raise SystemExit
else:
    print(f"Reading index file '{index_path}' took '{time.perf_counter() - tic:.3f}s'")

print(f"Reading '{len(index['files'])}' embeddings files...")
tic = time.perf_counter()
embeddings_folder = os.path.dirname(index_path)
embeddings_files = []
for file_name, _ in index['files']:
    toc = time.perf_counter()
    file_path = os.path.join(embeddings_folder, file_name)
    try:
        if ORJSON_AVAILABLE:
            with open(file_path, "rb") as f:
                embeddings = orjson.loads(f.read())
        else:
            with open(file_path, 'r') as f:
                embeddings = json.load(f)
    except Exception as e:
        print(f"Failed to read index '{file_path}': {e}")
        raise SystemExit
    else:
        embeddings_files.append(embeddings)
        print(f"Reading embeddings file '{file_path}' took '{time.perf_counter() - toc:.3f}s'")
print(f"Reading all '{len(index['files'])}' embeddings files took '{time.perf_counter() - tic:.3f}s'")

print("Unifying embeddings...")
tic = time.perf_counter()
cache = {}
for model in index['texts']:
    cache[model] = {}
    for text in index['texts'][model]:
        file_id = index['texts'][model][text][0]
        embedding_id = index['texts'][model][text][1]
        cache[model][text] = embeddings_files[file_id][embedding_id]
print(f"Unifying embeddings took '{time.perf_counter() - tic:.3f}s'")

target_folder = os.path.dirname(cache_path)
if not os.path.exists(target_folder):
    print(f"Creating target folder '{target_folder}'")
    os.makedirs(target_folder)

print(f"Writing cache file '{cache_path}'...")
tic = time.perf_counter()
try:
    if ORJSON_AVAILABLE:
        with open(cache_path, "wb") as f:
            f.write(orjson.dumps(cache, option=orjson.OPT_INDENT_2))
    else:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
except Exception as e:
    print(f"Failed to write cache file '{cache_path}': {e}")
    raise SystemExit
else:
    print(f"Writing cache file '{cache_path}' took '{time.perf_counter() - tic:.3f}s'")
