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

# cache file to read
cache_path = "C:\\Users\\Me\\Desktop\\cache_embeddings.json"
cache_path = "./../../cache/cache_embeddings.json"

# folder in which the the index and embeddings files are to be saved
embeddings_folder_path = "C:\\Users\\Me\\Desktop\\embeddings"
embeddings_folder_path = "./../../cache/embeddings"

# name of index file
index_name = "cache_embeddings_index.json"

# names of embeddings files
embeddings_name_template = "cache_embeddings_{file_id}.json"

# number of embeddings per embeddings files
embeddings_per_file = 100

### </Settings>

print(f"Reading cache file '{cache_path}'...")
tic = time.perf_counter()
try:
    if ORJSON_AVAILABLE:
        with open(cache_path, "rb") as f:
            cache = orjson.loads(f.read())
    else:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
except Exception as e:
    print(f"Failed to read cache file '{cache_path}': {e}")
    raise SystemExit
else:
    print(f"Reading cache file '{cache_path}' took '{time.perf_counter() - tic:.3f}s'")

print("Constructing index and splitting embeddings...'")
tic = time.perf_counter()
index = {'files': [], 'texts': {}}
next_file_id = 0
embeddings_files = []
for model in cache:
    print(f"Embeddings in model '{model}': {len(cache[model])}")
    floor_mod = (len(cache[model]) // embeddings_per_file, len(cache[model]) % embeddings_per_file)
    texts = sorted(cache[model])
    index['texts'][model] = {}
    for i in range(floor_mod[0]):
        file_name = embeddings_name_template.format(file_id=next_file_id)
        index['files'].append([file_name, embeddings_per_file])
        next_file_id += 1
        embeddings = []
        for j, text in enumerate(texts[embeddings_per_file * i: embeddings_per_file * (i + 1)]):
            index['texts'][model][text] = [len(index['files']) - 1, j]
            embeddings.append(cache[model][text])
        embeddings_files.append(embeddings)
    if floor_mod[1] > 0:
        file_name = embeddings_name_template.format(file_id=next_file_id)
        index['files'].append([file_name, floor_mod[1]])
        next_file_id += 1
        embeddings = []
        for j, text in enumerate(texts[embeddings_per_file * floor_mod[0]:]):
            index['texts'][model][text] = [len(index['files']) - 1, j]
            embeddings.append(cache[model][text])
        embeddings_files.append(embeddings)
print(f"Constructing index and splitting embeddings took '{time.perf_counter() - tic:.3f}s'")

if not os.path.exists(embeddings_folder_path):
    print(f"Creating target folder '{embeddings_folder_path}'")
    os.makedirs(embeddings_folder_path)

print("Writing new files...")

tic = time.perf_counter()
file_path = os.path.join(embeddings_folder_path, index_name)
try:
    if ORJSON_AVAILABLE:
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(index, option=orjson.OPT_INDENT_2))
    else:
        with open(file_path, 'w') as f:
            json.dump(index, f, indent=2)
except Exception as e:
    print(f"Failed to write index file '{file_path}': {e}")
    raise SystemExit
else:
    print(f"Writing index file '{file_path}' took '{time.perf_counter() - tic:.3f}s'")

tic = time.perf_counter()
for i, (embeddings_name, _) in enumerate(index['files']):
    toc = time.perf_counter()
    file_path = os.path.join(embeddings_folder_path, embeddings_name)
    try:
        if ORJSON_AVAILABLE:
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(embeddings_files[i]))
        else:
            with open(file_path, 'w') as f:
                json.dump(embeddings_files[i], f)
    except Exception as e:
        print(f"Failed to write embeddings file '{file_path}': {e}")
        raise SystemExit
    else:
        print(f"Writing embeddings file '{file_path}' took '{time.perf_counter() - toc:.3f}s'")
print(f"Writing all '{len(index['files'])}' embeddings files took '{time.perf_counter() - tic:.3f}s'")
