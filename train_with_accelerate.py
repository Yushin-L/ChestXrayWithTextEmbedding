# accelerate launch --multi_gpu --num_processes 4 /data/code/CXR_embedding_research/train_with_accelerate.py
import os
import pandas as pd
import numpy as np
import torch
import argparse
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from utils import custom_vit_embed, create_train_val_dataloaders, image_path_refine, KorniaGPUAugmentation
from transformers import ViTModel, ViTFeatureExtractor 
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
import torch
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CFG = {
    'Image_size':384,
    'EPOCHS':20,
    'MIN_LR':1e-6,
    'MAX_LR':3e-4,
    'LEARNING_RATE':3e-4,
    'SEED':42,
    'Train_BS':192,
    'Valid_BS':192,
    'optimizer':'AdamW',
    'scheduler':"CosineAnnealingLR",
    'model_name':"vit-large-patch16-384"
}

def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(seed):
    accelerator = Accelerator() # mixed_precision="bf16"
    set_seed(seed)
    mimic_train_df = pd.read_csv('/data/mimic3_cxr_jpg/train_with_view_embeddings.csv')
    mimic_train_df['ImagePath'] = mimic_train_df.apply(image_path_refine, axis=1)
    rex_train_df = pd.read_csv('/data/ReXGradient-160K/metadata/train_with_view_embeddings.csv')
    rex_train_df['ImagePath'] = rex_train_df['ImagePath'].apply(lambda x : x.replace('../', '/data/ReXGradient-160K/'))
    df_embed = pd.concat([mimic_train_df[['ImagePath', 'embeddings']],rex_train_df[['ImagePath', 'embeddings']]], axis=0).reset_index(drop=True)
    train_df, temp_df = train_test_split(df_embed, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, shuffle=True)
    train_df, val_df, test_df =  train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    processor = ViTFeatureExtractor.from_pretrained('/data/models/vit-base-patch16-384')
    train_loader, valid_loader = create_train_val_dataloaders(train_df, val_df, label_type="embedding", train_bs=CFG['Train_BS'], valid_bs=CFG['Valid_BS'], num_workers=20)
    augment_tool = KorniaGPUAugmentation().to('cuda')
    model = ViTModel.from_pretrained('/data/models/vit-base-patch16-384')
    model = custom_vit_embed(model)
    model = model.to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = CFG['LEARNING_RATE'], betas=(0.9,0.999), eps=1e-6, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=CFG['MIN_LR'], T_max=len(train_loader))

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    criterion = torch.nn.CosineEmbeddingLoss(reduction="sum")
    best_loss = 99
    df = pd.DataFrame(columns=['epoch','train_loss','valid_loss'])
    
    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        train_loss = []
        
        i = 0
        for imgs, labels in tqdm(iter(train_loader)):
            pixel_values = augment_tool(imgs, True)
            optimizer.zero_grad()
            output = model(pixel_values)
            ones = torch.ones(CFG['Train_BS']).to('cuda')
            loss = criterion(output, labels, ones)
            accelerator.backward(loss)
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            # i += 1
            # if i == 10:
            #     break
        _train_loss = np.mean(train_loss)
        valid_loss = []
        model.eval()
        i = 0
        with torch.no_grad():
            for imgs, labels in tqdm(iter(valid_loader)):
                pixel_values= augment_tool(imgs,False)
                output = model(pixel_values)
                ones = torch.ones(CFG['Valid_BS']).to('cuda')
                loss = criterion(output, labels, ones)
                valid_loss.append(loss.detach().cpu().numpy())
                # i += 1
                # if i == 10:
                #     break
                scheduler.step()
        _val_loss = np.mean(valid_loss)
        accelerator.print(
            f"Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}]]"
        )

        accelerator.wait_for_everyone()
        df = pd.concat([pd.DataFrame([[epoch,_train_loss,_val_loss]], columns=df.columns),df],ignore_index=True)
        df.to_csv(f'/data/code/CXR_embedding_research/history/vit-history-embed-{seed}-padding-aug-sum.csv',index=False)

        if best_loss >= _val_loss:
            best_loss = _val_loss
            accelerator.save_model(model, f"/data/mimic_ckp_models/vit-base-patch16-384-embedding-{seed}-padding-aug-sum.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main function with a specific seed.")
    parser.add_argument('--seed', type=int, required=True, help="Seed value for random number generation")
    args = parser.parse_args()
    train(args.seed)
