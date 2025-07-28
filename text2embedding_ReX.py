import os
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
import argparse
import json
from PIL import Image
from openai import OpenAI

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

def standardize_view_position_direct(df, column_name='ImageViewPosition'):
    """
    딕셔너리를 사용한 직접 매핑 방식
    """
    mapping = {
        'PA': 'PA',
        'POSTERO_ANTERIOR': 'PA',
        'AP': 'AP', 
        'ANTERO_POSTERIOR': 'AP',
        'AP AXIAL': 'AP'
    }
    
    df_standardized = df.copy()
    df_standardized[column_name] = df_standardized[column_name].map(mapping).fillna(df_standardized[column_name])
    return df_standardized

def text2embedding(client, model, text):
    responses = client.embeddings.create(
            input=[text],
            model=model,
        )
    return responses.data[0].embedding

def main():
    df = pd.read_csv('/data/ReXGradient-160K/metadata/train_metadata.csv')
    with open('/data/ReXGradient-160K/metadata/train_metadata_view_position.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df2 = process_multiple_medical_records(list(json_data.values()))
    df2 = standardize_view_position_direct(df2)
    df2 = df2[(df2['ImageViewPosition']=='AP') | (df2['ImageViewPosition']=='PA')]
    openai_api_key = "abc123"
    openai_api_base = "http://localhost:8002/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    embedding_rows = []
    for idx, row in tqdm(df2.iterrows()):
        note = "Findings: {} \nImpression: {}".format(row['Findings'], row['Impression'])
        embedding = text2embedding(client, model, note)
        embedding_rows.append(embedding)
    
    df2['embeddings'] = embedding_rows
    df2.to_csv('/data/ReXGradient-160K/metadata/train_with_view_embeddings.csv',encoding='utf8', index=False)


if __name__=='__main__':
    main()