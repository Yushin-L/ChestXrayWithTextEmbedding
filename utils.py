import os
import pandas as pd
import numpy as np
import torch
from PIL import ImageFile, Image
from torch.nn import functional as F
from tqdm import tqdm
import pydicom
import torchvision.transforms as transforms
import ast
import random
from kornia import augmentation as K

ImageFile.LOAD_TRUNCATED_IMAGES = True

#region Model architecture
class SimpleShallowBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = torch.nn.LayerNorm(768)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(768, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 4096)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)


# classifier : Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax
class SimpleClsBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = torch.nn.LayerNorm(768)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(768, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 13)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)

class SimpleClsBlock_rad_dino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = torch.nn.LayerNorm(768)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(768, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 13)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)

class SimpleTwoClsBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = torch.nn.LayerNorm(1024)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 2)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)


class custom_vit_embed(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = model
        self.projector = SimpleShallowBlock()
    def forward(self, pixel_values):
        output = self.vit(pixel_values)
        output = self.projector(output.pooler_output)
        return output


class custom_vit_vision_embed_two(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = model
        self.cls_layer = SimpleTwoClsBlock()
    def forward(self, pixel_values):
        output = self.vit(pixel_values)
        output = self.cls_layer(output.pooler_output)
        return output


class custom_vit_vision(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = model
        self.cls_layer = SimpleClsBlock()
    def forward(self, pixel_values):
        output = self.vit(pixel_values)
        output = self.cls_layer(output.pooler_output)
        return output

class custom_vit_vision_rad_dino(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = model
        self.cls_layer = SimpleClsBlock_rad_dino()
    def forward(self, pixel_values):
        output = self.vit(pixel_values)
        output = self.cls_layer(output.pooler_output)
        return output

#endregion

#region Datasets & Loader

def image_path_refine(row):
    return f'/data/mimic3_cxr_jpg/mimic-cxr-jpg-2.0.0.physionet.org/files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg'

class KorniaGPUAugmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
                
        # 개별 증강 모듈들
        self.color_jitter = K.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        )
        self.gaussian_blur = K.RandomGaussianBlur(
            kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5
        )
        self.gaussian_noise = K.RandomGaussianNoise(
            mean=0.0, std=0.01, p=0.3
        )
        self.gamma = K.RandomGamma(
            gamma=(0.8, 1.2), gain=(0.9, 1.1), p=0.3
        )
        self.normalize = K.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def forward(self, x: torch.Tensor, is_train:bool) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # 순차적으로 증강 적용
        with torch.no_grad():
            if is_train:
                x = self.gamma(x)
                x = self.color_jitter(x)
                x = self.gaussian_blur(x)
                x = self.gaussian_noise(x)
                x = self.normalize(x).detach()
            else:
                x = self.normalize(x).detach()
        return x.squeeze(0) if x.size(0) == 1 else x

class Embed_dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.ImagePath = df.ImagePath.values
        self.embeddings = df.embeddings
        # processor로부터 모델이 요구하는 최종 이미지 크기를 가져옵니다.
        
        self.cpu_transforms = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        ImagePath = self.ImagePath[index]
        embedding = self.embeddings[index]
        label = torch.tensor(ast.literal_eval(embedding))
        image = Image.open(ImagePath).convert('RGB')
        tensor_image = self.padding(image)
        tensor_image = self.cpu_transforms(tensor_image)
        return tensor_image, label
    
    def padding(self, img):
        """PIL Image를 정사각형으로 패딩하는 함수"""
        w, h = img.size
        max_size = max(w, h)
        
        # 새로운 정사각형 이미지 생성 (검은색 배경)
        padded_img = Image.new('RGB', (max_size, max_size), (0, 0, 0))
        
        # 중앙에 원본 이미지 붙여넣기
        paste_x = (max_size - w) // 2
        paste_y = (max_size - h) // 2
        padded_img.paste(img, (paste_x, paste_y))
        
        return padded_img

    def __len__(self):
        return len(self.ImagePath)


def create_dataloader(df, label_type='embedding', batch_size=32, shuffle=True, num_workers=4): # , augment=True
    """
    DataLoader 생성 함수
    
    Args:
        df: 이미지 경로와 임베딩이 포함된 DataFrame
        batch_size: 배치 크기 (기본값: 32)
        shuffle: 데이터 셔플 여부 (기본값: True)
        num_workers: 데이터 로딩 프로세스 수 (기본값: 4)
        augment: 데이터 증강 적용 여부 (기본값: True)
    
    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    
    # 데이터셋 생성
    if label_type == 'embedding':
        dataset = Embed_dataset(df) # , augment=augment
    
    # DataLoader 생성
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # GPU 사용 시 성능 향상
        drop_last=True    # 마지막 배치가 batch_size보다 작으면 제거
    )
    
    return dataloader

# 사용 예시
def create_train_val_dataloaders(train_df, val_df, label_type, augment=True, train_bs=32, valid_bs=64, num_workers=4):
    """
    훈련용과 검증용 DataLoader를 동시에 생성
    
    Args:
        train_df: 훈련 데이터 DataFrame
        val_df: 검증 데이터 DataFrame
        processor: 이미지 전처리를 위한 processor
        batch_size: 배치 크기
        num_workers: 데이터 로딩 프로세스 수
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    
    # 훈련용 DataLoader (augmentation 적용, shuffle=True)
    train_dataloader = create_dataloader(
        df=train_df,
        label_type=label_type,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 검증용 DataLoader (augmentation 미적용, shuffle=False)
    val_dataloader = create_dataloader(
        df=val_df,
        label_type=label_type,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_dataloader, val_dataloader


# class custom_test_dataset(torch.utils.data.Dataset):
#     def __init__(self, df, embeddings, processor):
#         self.dicom_ids = df.dicom_id.values
#         self.embed_ids = df.embed_index.values
#         self.text_ids = df.path.values
#         self.subject_ids = df.subject_id.values
#         self.study_ids = df.study_id.values
#         self.embedding_list = embeddings
#         self.processor = processor
#         self.classes = df.iloc[:,4:-4].values
#     def __getitem__(self,index):
#         dicom_ids = self.dicom_ids[index]
#         embed_ids = self.embed_ids[index]
#         text_ids = self.text_ids[index]
#         study_id = self.study_ids[index]
#         subject_id = self.subject_ids[index]
#         classes = self.classes[index]
#         img_path = f'/data/mimic3_cxr_jpg/mimic-cxr-jpg-2.0.0.physionet.org/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_ids}.jpg'
#         label = torch.tensor(self.embedding_list[embed_ids])
#         inputs = self.processor(Image.open(img_path).convert('RGB'), return_tensors='pt')
#         return inputs, label, img_path, text_ids, classes
#     def __len__(self):
#         return len(self.dicom_ids)


# class custom_dataset_cls(torch.utils.data.Dataset):
#     def __init__(self, df, processor):
#         self.dicom_ids = df.dicom_id.values
#         self.processor = processor
#         self.labels = df.iloc[:,4:-3].fillna(0).replace(-1, 0).values
#         self.subject_ids = df.subject_id.values
#         self.study_ids = df.study_id.values
#     def __getitem__(self,index):
#         dicom_ids = self.dicom_ids[index]
#         study_id = self.study_ids[index]
#         subject_id = self.subject_ids[index]
#         img_path = f'/data/mimic3_cxr_jpg/mimic-cxr-jpg-2.0.0.physionet.org/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_ids}.jpg'
#         label = self.labels[index]
#         inputs = self.processor(Image.open(img_path).convert('RGB'), return_tensors='pt')
#         return inputs, label, img_path
#     def __len__(self):
#         return len(self.dicom_ids)


# class Nih_dataset_cls(torch.utils.data.Dataset):
#     def __init__(self, df, processor):
#         self.dicom_ids = df['Image Index'].values
#         self.processor = processor
#         self.labels = df.iloc[:,-15:].values
#     def __getitem__(self,index):
#         dicom_ids = self.dicom_ids[index]
#         img_path = f'/home/imyousin12/datasets/nih_cxr_data/images/{dicom_ids}'
#         label = self.labels[index]
#         inputs = self.processor(Image.open(img_path).convert('RGB'), return_tensors='pt')
#         return inputs, label, img_path
#     def __len__(self):
#         return len(self.dicom_ids)


# class custom_dataset_two_cls(torch.utils.data.Dataset):
#     def __init__(self, df, processor):
#         self.dicom_ids = df.dicom_id.values
#         self.processor = processor
#         self.labels = df[['No Finding', 'Lung Opacity']].fillna(0).replace(-1, 0).values
#         self.subject_ids = df.subject_id.values
#         self.study_ids = df.study_id.values
#     def __getitem__(self,index):
#         dicom_ids = self.dicom_ids[index]
#         study_id = self.study_ids[index]
#         subject_id = self.subject_ids[index]
#         img_path = f'/data/mimic3_cxr_jpg/mimic-cxr-jpg-2.0.0.physionet.org/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_ids}.jpg'
#         label = self.labels[index]
#         inputs = self.processor(Image.open(img_path).convert('RGB'), return_tensors='pt')
#         return inputs, label, img_path
#     def __len__(self):
#         return len(self.dicom_ids)


# class Rsna_dataset(torch.utils.data.Dataset):
#     def __init__(self, df, file_path, processor):
#         self.df = df
#         self.file_path = file_path
#         self.processor = processor
#     def __getitem__(self,index):
#         data = self.df.iloc[index]
#         subject_id = data['StudyInstanceUID']
#         path = '/home/imyousin12/datasets/nih_cxr_data/images/{}'.format(self.file_path[subject_id])
#         label = torch.tensor(data['label_value'])
#         inputs = self.processor(Image.open(path).convert('RGB'), return_tensors='pt')
#         return inputs, label, path
#     def __len__(self):
#         return len(self.df)
    
# class Kaggle_dataset(torch.utils.data.Dataset):
#     def __init__(self, df, processor):
#         self.df = df
#         self.processor = processor
#     def __getitem__(self,index):
#         data = self.df.iloc[index]
#         path = data['Image_path']
#         label = data['Label']
#         if ".dcm" in path.split('/')[-1]:
#             inputs = self.processor(Image.fromarray(pydicom.dcmread(path).pixel_array).convert('RGB'),return_tensors='pt')
#         else:
#             inputs = self.processor(Image.open(path).convert('RGB'), return_tensors='pt')
#         return inputs, label, path
#     def __len__(self):
#         return len(self.df)

#endregion

