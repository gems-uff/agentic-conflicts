import pylcs
import os
import pandas as pd
import sys

def extract_candidate(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except:
        print('Error reading file: ', file_path)
        return ''

def analyze_folder(folder_path):
    files = os.listdir(folder_path)
    data = []
        
    for file in files:
        candidate = extract_candidate(folder_path+'/'+file)
        resolution = get_original_resolution(file)
        v1 = get_parent_v1(file)
        v2 = get_parent_v2(file)
        v1_similarity = calculate_similarity(candidate, v1)
        v2_similarity = calculate_similarity(candidate, v2)
        resolution_similarity = calculate_similarity(candidate, resolution)
        data.append([file, v1_similarity, v2_similarity])
    columns = ['chunk_id', 'candidate_v1_similarity', 'candidate_v2_similarity']    
    return pd.DataFrame(data, columns=columns)

def get_lcs(text1,text2):
    res = pylcs.lcs_sequence_idx(text1, text2)
    lcs = ''.join([text2[i] for i in res if i != -1])
    return lcs

def get_gestalt(file1_text,file2_text):
    try:
        if file1_text == file2_text:
            return 1
        return 2*len(get_lcs(file1_text, file2_text)) / (len(file1_text) + len(file2_text))
    except Exception as e:
        print(e)
        return 0

def get_original_resolution(chunk_id):
    return df_complete[df_complete['chunk_id']==chunk_id].iloc[0]['all_raw_res']

def get_parent_v1(chunk_id):
    return df_complete[df_complete['chunk_id']==chunk_id].iloc[0]['all_raw_a']
    
def get_parent_v2(chunk_id):
    return df_complete[df_complete['chunk_id']==chunk_id].iloc[0]['all_raw_b']

def calculate_similarity(output, reference):
    return get_gestalt(output, reference)


def main(dataset):
    path = f"../results/{dataset}/OUTPUT"
    df_result = analyze_folder(path)
    df_result.to_csv(f'../results/{dataset}/RESULTS/candidate_parents_similarities.csv', index=False)
   

execution_name = sys.argv[1]
df_complete = pd.read_json(f'../data/dataset2_Java_complete.json')
df_complete['chunk_id'] = df_complete['merge_id'].astype(str) + '-' + df_complete['chunk_number'].astype(str) 
main(execution_name)