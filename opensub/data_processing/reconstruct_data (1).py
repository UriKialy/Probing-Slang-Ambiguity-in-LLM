import xml.etree.ElementTree as ET
import string
import glob
import os
import numpy as np
import pandas as pd
import re

from tqdm import trange

# --- Utility regexes and constants ---
punctuations = set(string.punctuation)
re_multispace = re.compile(r"\s+")
# Match filenames like "12345.xml" (only digits, then ".xml")
re_xml = re.compile(r"^([0-9]+)\.xml$")

# use a raw string 
data_dir_prefix = r"C:\Users\yozev\Downloads\en (2)\OpenSubtitles\xml\en"


def parse_xml_file(xml_file: str):
    """
    Parse a single XML subtitle file. Returns a list of blocks, each with:
      { 'id': str, 'start_time': str, 'end_time': str, 'text': [word1, word2, ...] }
    """
    def _parse_s_block(iterator, s_block):
        while True:
            event, element = next(iterator)
            if element.tag == "time" and event == "start":
                time_id = element.attrib.get("id", "")
                time_val = element.attrib.get("value", "")
                # 'S1', 'E1', etc.
                if "S" in time_id:
                    s_block["start_time"] = time_val
                elif "E" in time_id:
                    s_block["end_time"] = time_val

            if element.tag == "w" and event == "start":
                # text node: element.text may be None or a word
                s_block["text"].append(element.text)

            if element.tag == "s" and event == "end":
                # end of this <s> block
                return s_block

    iterator = ET.iterparse(xml_file, events=["start", "end"])
    blocks = []

    while True:
        try:
            event, element = next(iterator)
            if element.tag == "s" and event == "start":
                s_block = {
                    "id": element.attrib.get("id", None),
                    "start_time": None,
                    "end_time": None,
                    "text": []
                }
                s_block = _parse_s_block(iterator, s_block)
                blocks.append(s_block)
        except StopIteration:
            break

    return blocks


def remove_space_before_punctuation(words: list[str]) -> list[str]:
    """
    Given a list of token strings, if a token begins with punctuation,
    append it directly to the previous token (no space).
    e.g. ["Hello", ",", "world", "!"] -> ["Hello,", "world!"]
    """
    if len(words) <= 1:
        return words

    new_list = [words[0]]
    for idx in range(1, len(words)):
        token = words[idx]
        if len(token) > 0 and token[0] in punctuations:
            # attach to previous
            new_list[-1] = new_list[-1] + token
        else:
            new_list.append(token)
    return new_list


def stringfy_block_with_idx_and_time(block: dict) -> str:
    """
    Convert one parsed <s> block into a string with:
      id = <id> start_time = <start> end_time = <end>
      <reconstructed text>
    """
    blk_id = block["id"]
    start_time = block["start_time"]
    end_time = block["end_time"]
    words = [w for w in block["text"] if w is not None]
    words = remove_space_before_punctuation(words)
    text_line = " ".join(words)

    return (
        f"id = {blk_id} start_time = {start_time} end_time = {end_time}\n"
        f"{text_line}\n"
    )


def process_file(in_file: str) -> str:
    """
    Parse one XML file, then stringify each <s> block.
    Return the entire file as one string (blocks joined by newline).
    """
    blocks = parse_xml_file(in_file)
    lines = [stringfy_block_with_idx_and_time(b) for b in blocks]
    return "\n".join(lines)


def lookup_sent(movie_id: str, sent_id: str, movie_sents: dict[str, np.ndarray]):
    """
    Given a movie_id and a sent_id, look up the three-line context around that sentence.
    If movie_id not in movie_sents or sent_id is out of range, return two empty strings.
    Otherwise return:
      ( sentence_string, context_string )
    where context_string has the form:
      "<previous> <i> <target> </i> <next>"
    """
    try:
        idx = int(sent_id)
    except ValueError:
        return "", ""

    if movie_id not in movie_sents:
        return "", ""

    arr = movie_sents[movie_id]
    # We want lines[idx-2], lines[idx-1], lines[idx] in zero-based Python
    # But the code originally used [sent_id-2 : sent_id+1], 1-based indexing.
    # So if sent_id = 5, zero-based idx=5, then the target is arr[4], context is arr[3], arr[4], arr[5].
    # To replicate original behavior, shift by one:
    zero_based = idx - 1
    if zero_based < 1 or zero_based + 1 >= len(arr):
        return "", ""

    prev_line = arr[zero_based - 1].strip()
    target_line = arr[zero_based].strip()
    next_line = arr[zero_based + 1].strip()
    context = f"{prev_line} <i> {target_line} </i> {next_line}"
    return target_line, context


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1) Read metadata TSVs
    # -------------------------------------------------------------------------
    data_meta = pd.read_csv(
        r"../../../../Downloads/slang_OpenSub_meta.tsv",
        dtype=str,
        sep="\t"
    ).fillna("")
    data_meta_vals = data_meta.values

    # Columns in data_meta (by index):
    #  0: some column,  1: some column, 2: MOVIE_ID, 3: SENT_ID, 4: some column, 5: YEAR, ...
    movie_ids = data_meta_vals[:, 2]
    sent_ids = data_meta_vals[:, 3]
    movie_years = data_meta_vals[:, 5]

    data_neg_meta = pd.read_csv(
        r"../../../../Downloads/slang_OpenSub_negatives_meta.tsv",
        dtype=str,
        sep="\t"
    ).fillna("")
    data_neg_vals = data_neg_meta.values

    movie_ids_neg = data_neg_vals[:, 0]
    sent_ids_neg = data_neg_vals[:, 1]

    movie_set = set(movie_ids)
    movie_list = sorted(movie_set)
    year_list = sorted({str(y) for y in movie_years if str(y).strip() != ""})

    # -------------------------------------------------------------------------
    # 2) Find all XML files on disk whose basename is "<digits>.xml"
    # -------------------------------------------------------------------------
    print("Finding XML files for relevant movies...")
    movie_files: dict[str, str] = {}

    for y in year_list:
        year_folder = os.path.join(data_dir_prefix, y)
        if not os.path.isdir(year_folder):
            continue

        # Each 'd' is a subfolder under ".../xml/en/<year>/"
        for d in glob.glob(os.path.join(year_folder, "*")):
            if not os.path.isdir(d):
                continue

            # Each 'f' is a file (e.g. "135737.xml") or another sub-subfolder
            for f in glob.glob(os.path.join(d, "*")):
                if not os.path.isfile(f):
                    continue
                fname = os.path.basename(f)  # e.g. "135737.xml"
                match = re_xml.match(fname)
                if match:
                    movie_id = match.group(1)  # e.g. "135737"
                    if movie_id in movie_set:
                        movie_files[movie_id] = f

    print("DONE")

    # -------------------------------------------------------------------------
    # 3) Extract sentences from each XML file, storing them in movie_sents[movie_id]
    # -------------------------------------------------------------------------
    print("Extracting sentences from movie subtitles...")
    movie_sents: dict[str, np.ndarray] = {}

    for m_id in trange(len(movie_list)):
        mid = movie_list[m_id]
        if mid not in movie_files:
            print(f"WARNING: no XML file on disk for movie_id = {mid}")
            continue

        xml_path = movie_files[mid]
        # process_file returns one big string; split by newline:
        content = process_file(xml_path)
        lines = content.split("\n")

        # We know each > block is 2 lines:
        #   1) "id = ... start_time = ... end_time = ..."
        #   2) "<text>"
        # Then a blank line before the next block, so effectively 3 lines per block:
        #   idx 0: id=... line
        #   idx 1: text line
        #   idx 2: "" (empty)
        # So to collect every second line (the text), we iterate i=1,4,7,... or simply skip i % 3 != 1
        all_sents = []
        for i in range(1, len(lines), 3):
            line = lines[i].strip()
            if line:
                # replace multiple whitespace with single space
                all_sents.append(re_multispace.sub(" ", line))

        movie_sents[mid] = np.array(all_sents, dtype=object)

    print("DONE")

    # -------------------------------------------------------------------------
    # 4) Reconstruct positive examples
    # -------------------------------------------------------------------------
    print("Reconstructing and saving full dataset...")

    data_sents = []
    data_contexts = []

    for idx in range(len(movie_ids)):
        mid = movie_ids[idx]
        sid = sent_ids[idx]
        sent_str, ctx_str = lookup_sent(mid, sid, movie_sents)
        data_sents.append(sent_str)
        data_contexts.append(ctx_str)

    data_sents = np.array(data_sents, dtype=object)
    data_contexts = np.array(data_contexts, dtype=object)

    # Build the final DataFrame for positives:
    # We want columns: ['SENTENCE', 'FULL_CONTEXT', 'SLANG_TERM', 'ANNOTATOR_CONFIDENCE',
    #                  'MOVIE_ID', 'SENT_ID', 'REGION', 'YEAR',
    #                  'DEFINITION_SENTENCE', 'DEFINITION_SOURCE_URL', 'LITERAL_PARAPHRASE_OF_SLANG']
    # Since data_meta_vals has exactly those extra columns (in that order), we can hstack:
    pos_combined = np.hstack([
        data_sents.reshape(-1, 1),
        data_contexts.reshape(-1, 1),
        data_meta_vals
    ])

    pos_columns = [
        "SENTENCE",
        "FULL_CONTEXT",
        "SLANG_TERM",
        "ANNOTATOR_CONFIDENCE",
        "MOVIE_ID",
        "SENT_ID",
        "REGION",
        "YEAR",
        "DEFINITION_SENTENCE",
        "DEFINITION_SOURCE_URL",
        "LITERAL_PARAPHRASE_OF_SLANG"
    ]

    output_pos = pd.DataFrame(pos_combined, columns=pos_columns)
    output_pos.to_csv("slang_OpenSub.tsv", sep="\t", index=False)

    # -------------------------------------------------------------------------
    # 5) Reconstruct negative examples
    # -------------------------------------------------------------------------
    data_neg_sents = []
    data_neg_contexts = []

    for idx in range(len(movie_ids_neg)):
        mid = movie_ids_neg[idx]
        sid = sent_ids_neg[idx]
        sent_str, ctx_str = lookup_sent(mid, sid, movie_sents)
        data_neg_sents.append(sent_str)
        data_neg_contexts.append(ctx_str)

    data_neg_sents = np.array(data_neg_sents, dtype=object)
    data_neg_contexts = np.array(data_neg_contexts, dtype=object)

    # Build the final DataFrame for negatives:
    # Columns: ['SENTENCE', 'FULL_CONTEXT', 'MOVIE_ID', 'SENT_ID', 'REGION', 'YEAR']
    neg_combined = np.hstack([
        data_neg_sents.reshape(-1, 1),
        data_neg_contexts.reshape(-1, 1),
        data_neg_vals
    ])

    neg_columns = [
        "SENTENCE",
        "FULL_CONTEXT",
        "MOVIE_ID",
        "SENT_ID",
        "REGION",
        "YEAR"
    ]

    output_neg = pd.DataFrame(neg_combined, columns=neg_columns)
    output_neg.to_csv("slang_OpenSub_negatives.tsv", sep="\t", index=False)

    print("DONE")
