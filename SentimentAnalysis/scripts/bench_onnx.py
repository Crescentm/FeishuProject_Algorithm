from typing import cast
import pandas as pd
import numpy as np
import time
import onnxruntime as ort
from transformers.models.bert import BertTokenizerFast
from sklearn.metrics import f1_score
from tqdm import tqdm

# 初始化 tokenizer
tokenizer = BertTokenizerFast.from_pretrained("./BertTokenizer")

# 标签映射
label2id = {'积极': 0, '中性': 2, '消极': 1}
id2label = {v: k for k, v in label2id.items()}

# 加载数据
df = pd.read_csv("data/base_val.csv")

# 加载 ONNX 模型
session = ort.InferenceSession("./model/minirbt_quant.onnx", providers=["CPUExecutionProvider"])
input_names = [inp.name for inp in session.get_inputs()]
output_name = session.get_outputs()[0].name

# 推理
all_preds = []
true_labels = []
total_time = 0

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["text"]
    label = row["label"]

    # Tokenize
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=128)
    ort_inputs = {name: inputs[name] for name in input_names}

    # 推理计时
    start = time.time()
    outputs = cast(
            list[np.ndarray],
            session.run([output_name], {k: inputs[k] for k in input_names}),
        )
    end = time.time()

    logits = outputs[0]
    pred = np.argmax(logits, axis=1)[0]

    all_preds.append(pred)
    true_labels.append(label2id[label])

    total_time += (end - start)

# 计算平均推理时间（毫秒）
avg_time_ms = (total_time / len(df)) * 1000
f1ma = f1_score(true_labels, all_preds, average='macro')
f1mi = f1_score(true_labels, all_preds, average="micro")

print(f"平均推理时间: {avg_time_ms:.2f} ms")
print(f"Macro F1 分数: {f1ma:.4f}")
print(f"Micro F1 分数: {f1mi:.4f}")
