import pandas as pd
import numpy as np
import time
import onnxruntime as ort
from transformers import BertTokenizerFast
from sklearn.metrics import f1_score
from tqdm import tqdm
import os

# 初始化 tokenizer
tokenizer = BertTokenizerFast.from_pretrained("hfl/minirbt-h256")

# 标签映射
label2id = {'积极': 0, '中性': 1, '消极': 2}
id2label = {v: k for k, v in label2id.items()}

# 加载数据
df = pd.read_csv("data/base_val.csv")

# 加载 ONNX 模型
session = ort.InferenceSession("minirbt_with_attention_quant.onnx", providers=["CPUExecutionProvider"])
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]  # ['logits', 'attentions']

# 推理
all_preds = []
true_labels = []
total_time = 0

save_attention_dir = "attention_output"
os.makedirs(save_attention_dir, exist_ok=True)

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["text"]
    label = row["label"]
    # 推理计时
    start = time.time()
    # Tokenize
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=128)
    ort_inputs = {name: inputs[name] for name in input_names}

    
    outputs = session.run(output_names, ort_inputs)
    

    logits = outputs[0]
    attentions = outputs[1]  # shape: [num_layers, 1, heads, seq_len, seq_len]

    pred = np.argmax(logits, axis=1)[0]
    end = time.time()
    all_preds.append(pred)
    true_labels.append(label2id[label])
    total_time += (end - start)

    # 可选：保存 attention 到文件
    #np.save(os.path.join(save_attention_dir, f"attn_{i}.npy"), attentions)

# 计算平均推理时间（毫秒）和 F1 分数
avg_time_ms = (total_time / len(df)) * 1000
f1 = f1_score(true_labels, all_preds, average='macro')

print(f"平均推理时间: {avg_time_ms:.2f} ms")
print(f"Macro F1 分数: {f1:.4f}")
