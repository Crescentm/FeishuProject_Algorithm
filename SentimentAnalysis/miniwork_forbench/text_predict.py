from typing import cast
import pandas as pd
import csv
import onnxruntime as ort
from transformers.models.bert import BertTokenizerFast
import numpy as np
from tqdm import tqdm

# load model
id2label = {0: "积极", 1: "消极", 2: "中性"}
tokenizer = BertTokenizerFast.from_pretrained("BertTokenizer")
session = ort.InferenceSession(
    "minirbt_quant.onnx", providers=["CPUExecutionProvider"]
)
input_names = [inp.name for inp in session.get_inputs()]
output_name = session.get_outputs()[0].name

# predict
df = pd.read_csv("base_val.csv", dtype=str, keep_default_na=False)
res = {"text": [], "label": []}

for _, item in tqdm(df.iterrows(), total=len(df)):
    text = item["text"]
    # Tokenize
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=128)
    ort_inputs = {name: inputs[name] for name in input_names}
    # 推理
    outputs = cast(
        list[np.ndarray],
        session.run([output_name], {k: inputs[k] for k in input_names}),
    )
    logits = outputs[0]
    score = np.max(logits, axis=1)[0] 
    label = np.argmax(logits, axis=1)[0]

    res["text"].append(item["text"])
    res["label"].append(id2label[label])

text_df = pd.DataFrame(data=res, dtype=str)
text_df.to_csv("text_predict.csv", index=False, quoting=csv.QUOTE_ALL)
