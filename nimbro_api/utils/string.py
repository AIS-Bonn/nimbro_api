#!/usr/bin/env python3

import re
import json

try:
    import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

def normalize(string, remove_underscores=False, remove_punctuation=False, remove_common_specials=False, remove_white_spaces=False, lowercase=False):
    if UNIDECODE_AVAILABLE:
        normalized = reduce_whitespace(remove_non_alpha_numeric(remove_unicode(string), remove_white_spaces=False, remove_underscores=remove_underscores, remove_punctuation=remove_punctuation, remove_common_specials=remove_common_specials, replace_by_space=True), remove_all=remove_white_spaces).strip()
    else:
        normalized = reduce_whitespace(remove_non_alpha_numeric(string, remove_white_spaces=False, remove_underscores=remove_underscores, remove_punctuation=remove_punctuation, remove_common_specials=remove_common_specials, replace_by_space=True), remove_all=remove_white_spaces).strip()
    if lowercase:
        return normalized.lower()
    else:
        return normalized

def remove_unicode(string):
    # Removes some unicode characters (like emojis or flags), and replaces others (like ä,ê,î,ł,æ...) by a normalized version (a,e,i,l,ae)
    return unidecode.unidecode(string).strip()

def reduce_whitespace(string, remove_all=False):
    # Reduces any number and combination of consecutive whitespaces (' ', '\t', '\n') by either replacing them with a single space or remving them entirely.
    if remove_all:
        return re.sub(r'\s+', '', string).strip()
    else:
        return re.sub(r'\s+', ' ', string).strip()

def remove_non_alpha_numeric(string, remove_white_spaces=True, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, replace_by_space=False):
    pattern = r"^a-zA-Z0-9"
    if not remove_white_spaces:
        pattern += " "
    if not remove_underscores:
        pattern += "_"
    if not remove_punctuation:
        pattern += ".,:!?"
    if not remove_common_specials:
        pattern += "()/$€&+-=*'<>;%"
    else:
        string = string.replace("'", "")
    if replace_by_space:
        return re.sub(f"[{pattern}]", " ", string).strip()
    else:
        return re.sub(f"[{pattern}]", "", string).strip()

def levenshtein_match(word, labels, threshold=0, normalization=False):
    if len(labels) == 0:
        return None

    distances = [levenshtein(word, label, normalization=normalization) for label in labels]
    min_i = distances.index(min(distances))

    if 0.0 < threshold < 1.0:
        threshold = int(len(word) * threshold) + 1

    if distances[min_i] <= threshold:
        return labels[min_i]
    else:
        return None

def levenshtein(a, b, normalization=False):
    if normalization:
        a = normalize(a, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, remove_white_spaces=True, lowercase=True)
        b = normalize(b, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, remove_white_spaces=True, lowercase=True)

    return levenshtein_raw(a, b)

def levenshtein_raw(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_raw(s2, s1)
    if not s1:
        return len(s2)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def extract_json_from_text(string, first_over_longest=False):
    opening_indices = []
    opening_indices.extend([m.start() for m in re.finditer(r'[{\[]', string)])
    if len(opening_indices) > 0:
        closing_indices = []
        closing_indices.extend([m.start() for m in re.finditer(r'[}\]]', string)])
        if len(closing_indices) > 0:
            options = []
            for i in opening_indices:
                for j in closing_indices:
                    if j < i:
                        continue
                    else:
                        options.append(string[i:j + 1])
            if not first_over_longest:
                options.sort(key=len, reverse=True)
            for option in options:
                try:
                    return json.loads(option)
                except json.JSONDecodeError:
                    pass
