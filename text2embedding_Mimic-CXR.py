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
import re

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

def text2embedding(client, model, text):
    responses = client.embeddings.create(
            input=[text],
            model=model,
        )
    return responses.data[0].embedding

def main():
    df = pd.read_csv('/data/mimic3_cxr_jpg/mimic-cxr-dataset.csv')
    df = standardize_view_position_direct(df)
    df = df[(df['ViewPosition'] == "PA") | (df['ViewPosition'] == "AP")].reset_index()    

    openai_api_key = "abc123"
    openai_api_base = "http://localhost:8002/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id

    embedding_rows = []
    for idx, row in tqdm(df.iterrows()):
        note = load_text('/data/mimic3_cxr_jpg/'+row['path'])
        note = text_processing(note)
        embedding = text2embedding(client, model, note)
        embedding_rows.append(embedding)
    
    df['embeddings'] = embedding_rows
    df.to_csv('/data/mimic3_cxr_jpg/train_with_view_embeddings.csv',encoding='utf8', index=False)


if __name__=='__main__':
    main()