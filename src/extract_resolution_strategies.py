import os
import json
import pandas as pd
from multiset import Multiset

class Chunk:
    def __init__(self, mergeId, language, fileName, repo, commitHash, chunk_index_merge,
                v1_size, v2_size, v1, v2, base, resolution, old_label, new_label): 
        self.mergeId = mergeId
        self.language = language
        self.file_name = fileName
        self.repo = repo
        self.commitHash = commitHash
        self.chunk_index_merge = chunk_index_merge
        self.v1_size = v1_size
        self.v2_size = v2_size
        self.v1 = v1
        self.v2 = v2
        self.base = base
        self.resolution = resolution
        self.old_label = old_label
        self.new_label = new_label
    
    def get_chunk_record_line(self):
        return [self.mergeId, self.language, self.file_name, self.repo, self.commitHash, self.chunk_index_merge,
               self.v1_size, self.v2_size, self.v1, self.v2, self.base, self.resolution, self.old_label, self.new_label]
    
def get_chunk_columns():
    return ['mergeId', 'language', 'fileName', 'repo', 'commitHash', 'chunk_index_merge',
            'v1_size', 'v2_size', 'v1', 'v2', 'base', 'resolution', 'old_label', 'new_label']

def extract_info_from_json(file_path, chunks, file_name, language):
    with open(file_path, 'r') as file:
        data = json.load(file)
        merge_id = file_name.split('_')[0]
        fname = data.get('fname', '')
        repo = data.get('repo', '')
        commit_hash = data.get('commitHash', '')
        conflicting_chunks = data.get('conflicting_chunks', [])
        chunk_index_merge = 1
        for chunk in conflicting_chunks:
            sizes = chunk.get('sizes', {}).get('lines', {})
            size_a = sizes.get('a', '')
            size_b = sizes.get('b', '')
            a_contents = chunk.get('a_contents', '')
            b_contents = chunk.get('b_contents', '')
            base_contents = chunk.get('base_contents', '')
            res_region = chunk.get('res_region', '')
            old_label = chunk.get('label', '')
            new_label = identify_resolution(a_contents, b_contents, res_region)
            if res_region != None:
                chunk_object = Chunk(merge_id, language, fname, repo, commit_hash, chunk_index_merge,
                             size_a, size_b, a_contents, b_contents, base_contents, res_region, old_label, new_label)
                chunks.append(chunk_object.get_chunk_record_line())
            chunk_index_merge+=1
            
def list_json_files(folder_path):
    chunks = []
    total_files = 1
    for root, dirs, files in os.walk(folder_path):
        language = os.path.basename(root)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                print("Processing:", file_path)
                extract_info_from_json(file_path, chunks, file, language)
                total_files+=1
    return pd.DataFrame(chunks, columns=get_chunk_columns())
                

def remove_empty_lines(lines):
    return [line for line in lines if line.strip()]

def identify_resolution(v1, v2, resolution):
    v1_lines = normalize_lines(remove_empty_lines(v1.splitlines()))
    v2_lines = normalize_lines(remove_empty_lines(v2.splitlines()))
    if resolution != None:
        resolution_lines = normalize_lines(remove_empty_lines(resolution.splitlines()))
        # Check for postponed resolution
        if '<<<<<<<' in resolution or '=======' in resolution or '>>>>>>>' in resolution:
            return 'Postponed'
    else:
        return 'Imprecise' # original dataset did not provide a resolution  

    if normalize_line(v1) == normalize_line(resolution):
        return 'V1'
    elif normalize_line(v2) == normalize_line(resolution):
        return 'V2'
    elif normalize_line(resolution) == normalize_line(v1) + normalize_line(v2):
        return 'ConcatV1V2'
    elif normalize_line(resolution) == normalize_line(v2) + normalize_line(v1):
        return 'ConcatV2V1'
    else:
        v1_ms = Multiset(v1_lines)
        v2_ms = Multiset(v2_lines)
        resolution_ms = Multiset(resolution_lines)
        if len(resolution_ms - (v1_ms + v2_ms)) > 0:
            return 'New code'
        elif len(resolution_lines) == 0:
            return 'None'
        else:
            return 'Combination'
        
def normalize_line(line):
    return line.replace(" ", "").replace("\t", "").replace("\n", "")

def normalize_lines(lines):
    normalized_lines = []
    for line in lines:
        normalized_lines.append(normalize_line(line))
    return normalized_lines

def save_dataset_to_json(dataset, file_name):
    json_data = dataset.to_dict(orient="list")
    with open(f"data/{file_name}", "w") as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == "__main__":
    folder_path = "data/fse2022/automated-analysis-data/Java"
    df1 = list_json_files(folder_path)
    df1['chunk_id'] = df1['mergeId'].astype(str) + '-' + df1['chunk_index_merge'].astype(str)
    folder_path = "data/fse2022/automated-analysis-data/CSharp"
    df2 = list_json_files(folder_path)
    df2['chunk_id'] = df2['mergeId'].astype(str) + '-' + df2['chunk_index_merge'].astype(str)
    folder_path = "data/fse2022/automated-analysis-data/JavaScript"
    df3 = list_json_files(folder_path)
    df3['chunk_id'] = df3['mergeId'].astype(str) + '-' + df3['chunk_index_merge'].astype(str)
    folder_path = "data/fse2022/automated-analysis-data/TypeScript"
    df4 = list_json_files(folder_path)
    df4['chunk_id'] = df4['mergeId'].astype(str) + '-' + df4['chunk_index_merge'].astype(str)
    df = pd.concat([df1, df2, df3, df4])
    save_dataset_to_json(df, 'dataset2_with_labels.json')
