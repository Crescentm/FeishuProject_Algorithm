import pandas as pd
import torch
import csv
import glob
from PIL import Image
from model import efficientnet_b7
from torchvision import transforms
from torchvision.transforms import InterpolationMode

model_path = "ckpts/best_model.pth"
model = efficientnet_b7(num_classes=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# load your model
model = model.cuda()
id2label = {0: 'cat', 1: 'dog', 2: 'other'}

transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

res = {'image_path': [], 'label': []}
for image_path in glob.glob('data/test/*'):
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda()
    scores = model(img)
    
    score, label = torch.max(scores, dim=-1)
    res['image_path'].append(image_path)
    res['label'].append(id2label[label.item()])

image_df = pd.DataFrame(data=res, dtype=str)
image_df.to_csv('res.csv', index=False, quoting=csv.QUOTE_ALL)