class x3ds_model(nn.Module):
    def __init__(self, num_classes=None):
        super(BaseModel2, self).__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).to(device) #2048
        self.base_model.blocks[5].proj = nn.Sequential(
            nn.Linear(2048, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(400, num_classes)
        ).to(device)
        
    def forward(self, x):        
        x = self.base_model(x)        
        return x
    
class x3dm_model(nn.Module):
    def __init__(self, num_classes=None):
        super(BaseModel3, self).__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True).to(device) #2048

        self.base_model.blocks[5].proj = nn.Sequential(
            nn.Linear(2048, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(400, num_classes)
        ).to(device)
        
    def forward(self, x):        
        x = self.base_model(x)        
        return x