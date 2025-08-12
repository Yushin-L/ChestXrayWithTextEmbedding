import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import argparse
import json
from openai import OpenAI
import re
import ast
from utils import *
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def standardize_view_position_direct(df, column_name='ViewPosition'):
    """
    딕셔너리를 사용한 직접 매핑 방식
    """
    mapping = {
        'PA': 'PA',
        'PA LLD': 'PA',
        'PA RLD': 'PA',
        'AP': 'AP', 
        'AP AXIAL': 'AP',
        'AP LLD': 'AP',
        'AP RLD': 'AP'
    }
    
    df_standardized = df.copy()
    df_standardized[column_name] = df_standardized[column_name].map(mapping).fillna(df_standardized[column_name])
    
    return df_standardized

def expand_medical_data_to_dataframe(data_dict):
    """
    의료 데이터 딕셔너리를 DataFrame으로 변환하면서 
    이미지 관련 정보를 개별 행으로 확장하는 함수
    Args:
        data_dict: 의료 데이터가 담긴 딕셔너리
        
    Returns:
        pd.DataFrame: 확장된 데이터프레임
    """
    # 이미지 관련 필드들 (리스트 형태로 되어있는 필드들)
    image_fields = ['ImagePath', 'ImageModality', 'ImageShape', 'ImageBodyPart', 'ImageViewPosition']
    # 이미지 개수 확인
    n_images = len(data_dict['ImagePath'])
    # 결과를 담을 리스트
    rows = []
    # 각 이미지에 대해 행을 생성
    for i in range(n_images):
        row = {}
        
        # 기본 정보들 (모든 행에 동일하게 복사)
        for key, value in data_dict.items():
            if key not in image_fields:
                row[key] = value
            else:
                # 이미지 관련 정보는 해당 인덱스의 값을 사용
                if isinstance(value, list) and i < len(value):
                    row[key] = value[i]
                else:
                    row[key] = None
        rows.append(row)
    # DataFrame 생성
    df = pd.DataFrame(rows)
    return df

def process_multiple_medical_records(data_list):
    """
    여러 의료 기록 딕셔너리를 처리하는 함수
    
    Args:
        data_list: 의료 데이터 딕셔너리들의 리스트
        
    Returns:
        pd.DataFrame: 모든 기록을 확장한 통합 데이터프레임
    """
    all_rows = []
    for data_dict in data_list:
        df = expand_medical_data_to_dataframe(data_dict)
        all_rows.append(df)
    # 모든 DataFrame을 하나로 합치기
    combined_df = pd.concat(all_rows, ignore_index=True)
    return combined_df

def load_text(path):
    with open(path,'r') as file:
        lines=file.readlines()
        file_content=''.join(lines)
    return file_content.split("FINAL REPORT\n ")[1].replace('\n ','\n') #

def text_processing(full_text):
    findings_pattern = r"FINDINGS:(.*?)"
    findings_match = re.search(findings_pattern, full_text, re.DOTALL)
    impression_pattern = r"IMPRESSION:(.*?)"
    impression_match = re.search(impression_pattern, full_text, re.DOTALL)
    if findings_match and impression_match:
        findings_start = findings_match.span()[0]
        impression_start = impression_match.span()[0]
        if findings_start <= impression_start :
            text = full_text[findings_start:]
        else:
            text = full_text[impression_start:]
    elif findings_match and (not impression_match):
        findings_start = findings_match.span()[0]
        text = full_text[findings_start:]
    elif (not findings_match) and impression_match:
        impression_start = impression_match.span()[0]
        text = full_text[impression_start:]
    else:
        text = full_text
    return text

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def make_completion(client, model, message)->str:
    completion = client.chat.completions.create(
            model=model,
            messages=message,
            reasoning_effort="low",
            temperature=0.8)
    return remove_think_tags(completion.choices[0].message.content)

def image_path_refine(row):
    return f'/data/mimic3_cxr_jpg/mimic-cxr-jpg-2.0.0.physionet.org/files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg'


paraphrase_prompt = '''
# Medical Report Paraphrasing System Prompt

You are a medical text paraphrasing assistant. Your task is to rephrase medical reports while maintaining identical clinical meaning and preserving all medical terminology.

## Core Instructions:

1. **Preserve Medical Terminology**: Keep all medical terms, anatomical references, and clinical vocabulary exactly as they appear (e.g., "consolidation", "pleural effusion", "pneumothorax", "cardiomediastinal silhouette")

2. **Maintain Clinical Accuracy**: The paraphrased text must convey the exact same medical information and clinical findings

3. **Rephrase Structure and Common Words**: Change sentence structure, connecting words, and non-medical vocabulary while keeping the clinical content intact

4. **Output Format**: Return the result as a JSON object with the key "paraphrased_note"

## Paraphrasing Guidelines:

### What TO Change:
- Sentence structure and word order
- reorder findings naturally while maintaining logical flow
- Common verbs (e.g., "There is" → "Shows", "are noted" → "is observed")
- Descriptive phrases (e.g., "most likely represent" → "are likely due to")
- Connecting words and prepositions
- Passive vs active voice when appropriate

### What NOT TO Change:
- Medical terminology and clinical terms
- Anatomical references (rib numbers, organ names, etc.)
- Numerical values and measurements
- Section headers (FINDINGS, IMPRESSION)

## Output Requirements:
- Return only a JSON object with "paraphrased_note" as the key
- Maintain the original section structure (FINDINGS and IMPRESSION)
- Ensure all medical findings are preserved with identical clinical meaning
- Use the exact JSON format shown below:
{
  "paraphrased_note": "generated note"
}
'''

evaluation_prompt = '''
# Medical Report Paraphrasing Evaluation Criteria

## Evaluation Criteria:

### 1. Medical Terminology Preservation (40 points)
**Excellent (40 points)**: All medical terms, anatomical references, and clinical vocabulary are preserved exactly with zero alterations
- Complete preservation of all medical terminology

**Fair (20 points)**: Some medical terminology altered but core clinical meaning preserved
- 3-5 medical terms modified with synonyms
- Minor variations in non-critical medical terms

**Poor (0 points)**: Significant medical terminology alterations that could affect clinical interpretation
- Multiple critical medical terms changed
- Essential medical vocabulary lost or modified

### 2. Clinical Accuracy & Information Completeness (40 points)
**Excellent (40 points)**: All clinical findings, diagnoses, and medical information are identical in meaning with complete preservation
- All specific findings mentioned (clips, rib deformity, nipple shadows, etc.)
- Diagnostic conclusions preserved exactly
- No clinical information lost, added, or modified

**Fair (20 points)**: Most clinical information preserved with minor omissions or additions that don't affect diagnosis
- 1-2 minor clinical details may be slightly altered
- Core diagnostic meaning maintained

**Poor (0 points)**: Significant clinical information altered, lost, added, or modified in meaning
- Important findings omitted or incorrectly stated
- Diagnostic conclusions changed

### 3. Structural and Linguistic Paraphrasing (10 points)
**Excellent (10 points)**: Exceptional variation in sentence structure and non-medical vocabulary
- Complete reordering of sentences within sections
- Creative and natural use of synonyms for all non-medical words

**Fair (5 points)**: Moderate structural changes with some vocabulary variation
- Some sentences restructured
- Limited but appropriate use of synonyms

**Poor (0 points)**: Minimal structural changes or mostly identical phrasing to original
- Little to no sentence restructuring
- Minimal vocabulary variation

### 4. Format and Organization Compliance (10 points)
**Excellent (10 points)**: Flawless adherence to format requirements
- Maintains FINDINGS and IMPRESSION sections exactly
- Perfect logical flow of information

**Fair (5 points)**: Adequate format compliance with some structural issues
- Sections present but may have minor formatting inconsistencies

**Poor (0 points)**: Significant format violations, missing sections, or unprofessional presentation
- Incorrect or missing section headers
- Poor organization or inappropriate length

## Grade Assignment:

**Good**: 75 or higher points totally. **include 75 points
**Bad**: Under 75 points totally.

## Output Requirements:
- Return only a JSON object with "grade" as the key
- Use the exact JSON format shown below:
{
  "grade": "Good or Bad"
}

'''

def paraphrase_func(client, model, txt, p_prompt, e_prompt, retry=5):
    p_message = [
        {"role": "user", "content": "<system prompt>\n{}\n</system prompt>\n\n <clinical note>\n{}\n</clinical note>".format(p_prompt, txt)}
        ]
    paraphrase_completion = tojson(make_completion(client, model=model, message=p_message))
    e_message = [
        {"role": "user", "content": "<system prompt>\n{}\n</system prompt>\n\n<original note>\n{}\n</original note>\n<generation note>\n{}\n</generation note>".format(e_prompt, txt, paraphrase_completion)}
        ]
    evaluate_completion = tojson(make_completion(client, model=model, message=e_message))
    if evaluate_completion['grade'] == "Good":
        return paraphrase_completion
    else:
        retry -= 1
        if retry == 0:
            return {"paraphrased_note":"Fail"}
        return paraphrase_func(client, model, txt, p_prompt, e_prompt, retry=retry)

def tojson(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        dict_str = match.group()
        data = ast.literal_eval(dict_str)
        return data
    else:
        text

openai_api_key = "sk-1234"
openai_api_base = "http://localhost:4000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# def main():
#     # 로깅 설정
#     logging.basicConfig(
#         filename='/data/code/CXR_embedding_research/logs.txt',
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     mimic_train_df = pd.read_csv('/data/mimic3_cxr_jpg/mimic-cxr-dataset.csv')
#     mimic_train_df['ImagePath'] = mimic_train_df.apply(image_path_refine, axis=1)
#     mimic_train_df = standardize_view_position_direct(mimic_train_df)
#     mimic_train_df = mimic_train_df[(mimic_train_df['ViewPosition'] == "PA") | (mimic_train_df['ViewPosition'] == "AP")].reset_index()
#     mimic_train_df['paraphrased_note'] = ''
    
#     total_rows = len(mimic_train_df)
#     logging.info(f"시작 - 총 {total_rows}개 행 처리 예정")
    
#     for idx, row in mimic_train_df.iterrows():
#         start_time = datetime.now()
#         logging.info(f"처리 시작 - idx: {idx}")
#         try:
#             original_path = "/data/mimic3_cxr_jpg/" + row['path']
#             txt = text_processing(load_text(original_path))
#             paraphrase_completion = paraphrase_func(client, 'gpt-oss-120b', txt, paraphrase_prompt, evaluation_prompt)
#             mimic_train_df.at[idx, 'paraphrased_note'] = paraphrase_completion['paraphrased_note']
                
#             # 같은 디렉토리에 aug- 접두사로 저장
#             dir_name = os.path.dirname(original_path)
#             file_name = "aug-" + os.path.basename(row['path'])
#             aug_path = os.path.join(dir_name, file_name)
            
#             # 텍스트 파일로 저장
#             with open(aug_path, 'w', encoding='utf-8') as f:
#                 f.write(paraphrase_completion['paraphrased_note'])
            
#             end_time = datetime.now()
#             duration = (end_time - start_time).total_seconds()
#             logging.info(f"처리 완료 - idx: {idx}, 소요시간: {duration:.2f}초")
            
#         except Exception as e:
#             original_path = "/data/mimic3_cxr_jpg/" + row['path']
#             dir_name = os.path.dirname(original_path)
#             file_name = "aug-" + os.path.basename(row['path'])
#             aug_path = os.path.join(dir_name, file_name)

#             end_time = datetime.now()
#             duration = (end_time - start_time).total_seconds()
#             logging.error(f"처리 실패 - idx: {idx}, 소요시간: {duration:.2f}초, 오류: {str(e)}")
            
#             with open(aug_path, 'w', encoding='utf-8') as f:
#                 f.write("Fail")
        
#     logging.info("전체 처리 완료")

log_lock = threading.Lock()

# # MIMIC
# def process_single_row(args):
#     """단일 행을 처리하는 함수"""
#     idx, row, client = args
#     start_time = datetime.now()
    
#     try:
#         original_path = "/data/mimic3_cxr_jpg/" + row['path']
#         txt = text_processing(load_text(original_path))
#         paraphrase_completion = paraphrase_func(client, 'gpt-oss-120b', txt, paraphrase_prompt, evaluation_prompt)
#         paraphrased_note = paraphrase_completion['paraphrased_note']
            
#         # 같은 디렉토리에 aug- 접두사로 저장
#         dir_name = os.path.dirname(original_path)
#         file_name = "aug-" + os.path.basename(row['path'])
#         aug_path = os.path.join(dir_name, file_name)
        
#         # 텍스트 파일로 저장
#         with open(aug_path, 'w', encoding='utf-8') as f:
#             f.write(paraphrased_note)
        
#         end_time = datetime.now()
#         duration = (end_time - start_time).total_seconds()
        
#         with log_lock:
#             logging.info(f"처리 완료 - idx: {idx}, file-name: {file_name}, 소요시간: {duration:.2f}초, Thread: {threading.current_thread().name}")
        
#         return idx, paraphrased_note, True, None
        
#     except Exception as e:
#         original_path = "/data/mimic3_cxr_jpg/" + row['path']
#         dir_name = os.path.dirname(original_path)
#         file_name = "aug-" + os.path.basename(row['path'])
#         aug_path = os.path.join(dir_name, file_name)

#         end_time = datetime.now()
#         duration = (end_time - start_time).total_seconds()
        
#         with log_lock:
#             logging.error(f"처리 실패 - idx: {idx}, 소요시간: {duration:.2f}초, 오류: {str(e)}, Thread: {threading.current_thread().name}")
        
#         with open(aug_path, 'w', encoding='utf-8') as f:
#             f.write("Fail")
            
#         return idx, "Fail", False, str(e)

# ReX-CXR
def process_single_row(args):
    """단일 행을 처리하는 함수"""
    idx, row, client = args
    start_time = datetime.now()
    
    try:
        txt = "Findings: {} \nImpression: {}".format(row['Findings'], row['Impression'])
        paraphrase_completion = paraphrase_func(client, 'gpt-oss-120b', txt, paraphrase_prompt, evaluation_prompt)
        paraphrased_note = paraphrase_completion['paraphrased_note']
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        with log_lock:
            logging.info(f"처리 완료 - idx: {idx}, rid: {row['PatientID']}, 소요시간: {duration:.2f}초, Thread: {threading.current_thread().name}")
        
        return idx, paraphrased_note, True, None
        
    except Exception as e:
        
        with log_lock:
            logging.error(f"처리 실패 - idx: {idx}, 소요시간: {duration:.2f}초, 오류: {str(e)}, Thread: {threading.current_thread().name}")
            
        return idx, "Fail", False, str(e)


# def main_mimic():
#     # 로깅 설정
#     logging.basicConfig(
#         filename='/data/code/CXR_embedding_research/logs.txt',
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     mimic_train_df = pd.read_csv('/data/mimic3_cxr_jpg/mimic-cxr-dataset.csv')
#     mimic_train_df['ImagePath'] = mimic_train_df.apply(image_path_refine, axis=1)
#     mimic_train_df = standardize_view_position_direct(mimic_train_df)
#     mimic_train_df = mimic_train_df[(mimic_train_df['ViewPosition'] == "PA") | (mimic_train_df['ViewPosition'] == "AP")].reset_index()
#     mimic_train_df['paraphrased_note'] = ''
    
#     total_rows = len(mimic_train_df)
#     logging.info(f"시작 - 총 {total_rows}개 행 처리 예정 (8개 스레드 사용)")
    
#     # 각 스레드에서 사용할 인자들 준비
#     process_args = [(idx, row, client) for idx, row in mimic_train_df.iterrows()]
    
#     # ThreadPoolExecutor로 8개 스레드 사용
#     with ThreadPoolExecutor(max_workers=64) as executor:
#         # 모든 작업 제출
#         future_to_idx = {executor.submit(process_single_row, args): args[0] for args in process_args}
        
#         completed_count = 0
        
#         # 완료된 작업들을 처리
#         for future in as_completed(future_to_idx):
#             try:
#                 idx, paraphrased_note, success, error = future.result()
                
#                 # DataFrame에 결과 저장
#                 mimic_train_df.at[idx, 'paraphrased_note'] = paraphrased_note
                
#                 completed_count += 1
                
#                 if completed_count % 1000 == 0:  # 50개마다 진행상황 로그
#                     with log_lock:
#                         logging.info(f"진행상황: {completed_count}/{total_rows} 완료 ({completed_count/total_rows*100:.1f}%)")
                        
#             except Exception as e:
#                 idx = future_to_idx[future]
#                 with log_lock:
#                     logging.error(f"Future 처리 중 오류 발생 - idx: {idx}, 오류: {str(e)}")
    
#     logging.info(f"모든 처리 완료 - 총 {total_rows}개 행")
    
#     # 결과 DataFrame 저장 (옵션)
#     output_path = '/data/code/CXR_embedding_research/processed_dataset.csv'
#     mimic_train_df.to_csv(output_path, index=False)
#     logging.info(f"결과 저장 완료: {output_path}")

def main_rex():
    # 로깅 설정
    logging.basicConfig(
        filename='/data/code/CXR_embedding_research/logs.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    with open('/data/ReXGradient-160K/metadata/train_metadata_view_position.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    rex_train_df = process_multiple_medical_records(list(json_data.values()))
    rex_train_df = standardize_view_position_direct(rex_train_df, column_name= 'ImageViewPosition')
    rex_train_df = rex_train_df[(rex_train_df['ImageViewPosition']=='AP') | (rex_train_df['ImageViewPosition']=='PA')]    
    total_rows = len(rex_train_df)
    logging.info(f"시작 - 총 {total_rows}개 행 처리 예정 (8개 스레드 사용)")
    
    # 각 스레드에서 사용할 인자들 준비
    process_args = [(idx, row, client) for idx, row in rex_train_df.iterrows()]
    
    # ThreadPoolExecutor로 8개 스레드 사용
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 모든 작업 제출
        future_to_idx = {executor.submit(process_single_row, args): args[0] for args in process_args}
        
        completed_count = 0
        
        # 완료된 작업들을 처리
        for future in as_completed(future_to_idx):
            try:
                idx, paraphrased_note, success, error = future.result()
                
                # DataFrame에 결과 저장
                rex_train_df.at[idx, 'paraphrased_note'] = paraphrased_note
                
                completed_count += 1
                
                if completed_count % 1000 == 0:  # 50개마다 진행상황 로그
                    with log_lock:
                        logging.info(f"진행상황: {completed_count}/{total_rows} 완료 ({completed_count/total_rows*100:.1f}%)")
                        
            except Exception as e:
                idx = future_to_idx[future]
                with log_lock:
                    logging.error(f"Future 처리 중 오류 발생 - idx: {idx}, 오류: {str(e)}")
    
    logging.info(f"모든 처리 완료 - 총 {total_rows}개 행")
    
    # 결과 DataFrame 저장 (옵션)
    output_path = '/data/code/CXR_embedding_research/processed_dataset_rex.csv'
    rex_train_df.to_csv(output_path, index=False)
    logging.info(f"결과 저장 완료: {output_path}")

if __name__=="__main__":
    main_rex()