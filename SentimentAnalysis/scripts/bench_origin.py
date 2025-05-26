# 平均推理时间: 585.10 ms
import torch
import time
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from transformers import logging,BertTokenizerFast,BertModel
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score
logging.set_verbosity_error()

class SentimentDataset(Dataset):
    def __init__(self, csv_file, tokenizer, label2id, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = self.label2id[self.data.iloc[idx]['label']]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }

class MiniRBTClassifier(nn.Module):
    def __init__(self, model_name="hfl/minirbt-h256", num_labels=3):
        super(MiniRBTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits  # shape: [batch_size, num_labels]

tokenizer = BertTokenizerFast.from_pretrained("hfl/minirbt-h256")
model = MiniRBTClassifier(model_name="hfl/minirbt-h256")  # 重新实例化模型

# 加载模型权重
model.load_state_dict(torch.load('minirbt_sentiment_model.pth',map_location='cpu',weights_only=False))
# 确保模型在 CPU 上
device = torch.device("cpu")
model.to(device)
model.eval()

# 标签映射
label2id = {'积极': 0, '中性': 1, '消极': 2}
id2label = {v: k for k, v in label2id.items()}

# 加载数据
df = pd.read_csv("data/base_val.csv")
tokenizer = BertTokenizerFast.from_pretrained('hfl/minirbt-h256')


# 推理
all_preds = []
true_labels = []
total_time = 0
max_length=512

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["text"]
    label = row["label"]

    # Tokenize
    start = time.time()
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=128)
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # 进行推理
    with torch.no_grad():  # 关闭梯度计算以节省内存
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    # 获取模型的预测结果
    predicted_class = torch.argmax(output, dim=1).item()
    end = time.time()

    all_preds.append(predicted_class)
    true_labels.append(label2id[label])

    total_time += (end - start)

# 计算平均推理时间（毫秒）
avg_time_ms = (total_time / len(df)) * 1000
f1 = f1_score(true_labels, all_preds, average='macro')

print(f"平均推理时间: {avg_time_ms:.2f} ms")
print(f"Macro F1 分数: {f1:.4f}")
