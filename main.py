import random
import pandas as pd
import numpy as np
import os
import cv2 as cv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

#augmentation
#from pytorchvideo.transforms.transforms_factory import create_video_transform
#from transformers import AutoModel, AutoImageProcessor, AutoConfig

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="car-blackbox_label3",
    
    # track hyperparameters and run metadata
    config={
    'video_length' : 50, #10프레임 5초
    'img_size' : 128,
    'epoch' : 5,
    'learning_rate' : 3e-4,
    'batch_size' : 64,
    'seed' : 2023
    }
)

device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'video_length' : 50, #10프레임 5초
    'img_size' : 128,
    "model_name":"facebook/timesformer-base-finetuned-k400",
    'epoch' : 10,
    'learning_rate' : 3e-4,
    'batch_size' : 64,
    'seed' : 2023
}
#wandb.config.updata(CFG)

def seed_everything(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
#seed 고정
seed_everything(CFG['seed'])

info_to_label = {'100':1,'101':2,'110':3,'111':4,'120':5,'121':6,'200':7,'201':8,'210':9,'211':10,'220':11,'221':12}
label_to_info = {1:'100',2:'101',3:'110',4:'111',5:'120',6:'121',7:'200',8:'201',9:'210',10:'211',11:'220',12:'221'}

def get_label_info(label):
    if label == 0:
        crash_ego_label, weather_label, timing_label = 0, np.NaN, np.NaN
    else:
        infos = list(map(int, list(label_to_info[label])))
        crash_ego_label, weather_label, timing_label = infos[0], infos[1], infos[2]
    return crash_ego_label, weather_label, timing_label

df = pd.read_csv('/data2/dlstj/dacon_carvideo_classification/train.csv')
df['video_path'] = df['video_path'].replace(to_replace='./train', value='/data2/dlstj/dacon_carvideo_classification/train', regex=True)

crash_ego_label = []
weather_label = []
timing_label = []
for label in df['label']:
    res1, res2, res3 = get_label_info(label)
    crash_ego_label.append(res1)
    weather_label.append(res2)
    timing_label.append(res3)
    
df['crash_ego_labels'] = crash_ego_label
df['weather_labels'] = weather_label
df['timing_labels'] = timing_label

#train,val, _, _ = train_test_split(df, df['label'], test_size=0.2,random_state=CFG['seed'])
df_crash_ego = df
df_weather = df[df['weather_labels'].notna()].reset_index(drop=True)
df_weather = df_weather[df_weather['crash_ego_labels'] != 0] # crash 인 경우 weather 데이터만 사용
df_timing = df[df['timing_labels'].notna()].reset_index(drop=True)
df_timing = df_timing[df_timing['crash_ego_labels'] != 0] # crash 인 경우 timing 데이터만 사용



class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, transforms=None):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, idx):
        frames = self.get_video(self.video_path_list[idx])
        
        if self.transforms is not None:
            res = self.transforms(**frames)
            images = torch.zeros((len(images), 3, CFG["IMG_SIZE"], CFG["IMG_SIZE"]))
            images[0, :, :, :] = res["image"]
            for i in range(1, len(images)):
                images[i, :, :, :] = res[f"image{i}"]
        
        if self.label_list is not None:
            label = self.label_list[idx]
            return frames, label
        else:
            return frames
    
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self,path):
        frames = []
        cap = cv.VideoCapture(path)
        for _ in range(CFG['video_length']):
            _, img = cap.read()
            try:
                img = cv.resize(img,(128,128))
            except:
                break
            img = img/255.
            frames.append(img)
        # (batch, channel,height,width) -> ()
        return torch.FloatTensor(np.array(frames)).permute(3,0,1,2)
    
class crash_ego_Model(nn.Module):
    def __init__(self, pretrained=True):
        super(crash_ego_Model, self).__init__()
        self.feature_extract = models.video.r3d_18(pretrained=pretrained)
        self.crash_ego = nn.Sequential(
                                        nn.Linear(400, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 3),
        )

        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        crash_ego = self.crash_ego(x)
        return crash_ego
    
class weather_Model(nn.Module):
    def __init__(self, pretrained=True):
        super(weather_Model, self).__init__()
        self.feature_extract = models.video.r3d_18(pretrained=pretrained)
        self.weather = nn.Sequential(
                                        nn.Linear(400, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 3),
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        weather = self.weather(x)
        return weather
    
class timing_Model(nn.Module):
    def __init__(self, pretrained=True):
        super(timing_Model, self).__init__()
        self.feature_extract = models.video.r3d_18(pretrained=pretrained)
        self.timing = nn.Sequential(
                                        nn.Linear(400, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 2),
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        timing = self.timing(x)
        return timing
    
#data imbalance에 좋은 loss 사용

class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
    
def train(model, optimizer, train_loader, val_loader, scheduler, device, label_name):
    model.to(device)
    criterion = ASLSingleLabel().to(device)
    
    best_val_score = 0
    best_model = None
    
    for i, epoch in enumerate(range(1, CFG['epoch']+1)):
        model.train()
        train_loss = []
        
        for j ,(videos, labels) in enumerate(tqdm(iter(train_loader))):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(videos)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
        _val_loss, _val_score = validation(model, criterion, val_loader, device, label_name)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        wandb.log({"train_loss": _train_loss, "validation_loss":_val_loss, "F1_score":_val_score}, step=epoch)
        
        if scheduler is not None:
            scheduler.step(_val_score)
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            print('best_f1_score', best_val_score)
            
    return best_model

def validation(model, criterion, val_loader, device, label_name):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            
        _val_loss = np.mean(val_loss)
        
    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

train_crash_ego, val_crash_ego, _, _ = train_test_split(df_crash_ego, df_crash_ego['crash_ego_labels'], test_size=0.2, random_state=CFG['seed'])
train_weather, val_weather, _, _ = train_test_split(df_weather, df_weather['weather_labels'], test_size=0.2, random_state=CFG['seed'])
train_timing, val_timing, _, _ = train_test_split(df_timing, df_timing['timing_labels'], test_size=0.2, random_state=CFG['seed'])

train_crash_ego_dataset = CustomDataset(train_crash_ego['video_path'].values, train_crash_ego['crash_ego_labels'].values)
train_crash_ego_loader = DataLoader(train_crash_ego_dataset, batch_size = CFG['batch_size'], shuffle=True, num_workers=0)
val_crash_ego_dataset = CustomDataset(val_crash_ego['video_path'].values, val_crash_ego['crash_ego_labels'].values)
val_crash_ego_loader = DataLoader(val_crash_ego_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=0)

train_weather_dataset = CustomDataset(train_weather['video_path'].values, train_weather['weather_labels'].values)
train_weather_loader = DataLoader(train_weather_dataset, batch_size = CFG['batch_size'], shuffle=True, num_workers=0)
val_weather_dataset = CustomDataset(val_weather['video_path'].values, val_weather['weather_labels'].values)
val_weather_loader = DataLoader(val_weather_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=0)

train_timing_dataset = CustomDataset(train_timing['video_path'].values, train_timing['timing_labels'].values)
train_timing_loader = DataLoader(train_timing_dataset, batch_size = CFG['batch_size'], shuffle=True, num_workers=0)
val_timing_dataset = CustomDataset(val_timing['video_path'].values, val_timing['timing_labels'].values)
val_timing_loader = DataLoader(val_timing_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=0)

# crash_ego model 학습
model_crashego = crash_ego_Model()
optimizer = torch.optim.AdamW(params = model_crashego.parameters(), lr = CFG["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
infer_model_crashego = train(model_crashego, optimizer, train_crash_ego_loader, val_crash_ego_loader, scheduler, device, 'crash_ego')

model_weather = weather_Model()
infer_model_weather = train(model_weather, optimizer, train_weather_loader, val_weather_loader, scheduler, device, 'weather')

model_timing = timing_Model()
infer_model3 = train(model_timing, optimizer, train_timing_loader, val_timing_loader, scheduler, device, 'timing')