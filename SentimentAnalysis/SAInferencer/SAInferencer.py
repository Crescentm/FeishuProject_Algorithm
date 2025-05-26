from typing import cast
import numpy as np
import onnxruntime as ort
from transformers.models.bert import BertTokenizerFast
from sklearn.metrics import f1_score
import re
import json


class SAInferencer:

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        label_map: dict,
        max_length: int = 512,
        valid_keywords: str = "valid_keywords.json",
    ):
        self.label2id = label_map
        self.id2label = {v: k for k, v in label_map.items()}
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.max_length = max_length
        self.valid_keywords = json.load(
            open(valid_keywords, "r", encoding="utf-8")
        )["valid_keywords"]

    def clean_text(self, text):

        if not isinstance(text, str):
            return ""

        def filter_emoji(match):
            emoji = match.group(0)  # 带中括号的表情
            content = emoji[1:-1]  # 去掉中括号
            if any(keyword in content for keyword in self.valid_keywords):
                return emoji
            else:
                return ""

        text = re.sub(r"\[[^\[\]]+\]", filter_emoji, text)
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"//@\S+?:", "", text)
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"(\[[^\[\]]+\])\1+", r"\1", text)
        text = re.sub(r"([！？!。，、,.，?])\1+", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("转发微博", "")

        return text

    def predict(self, text: str) -> str:
        text = self.clean_text(text)
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        outputs: list[np.ndarray] = cast(
            list[np.ndarray],
            self.session.run(output_names, {k: inputs[k] for k in input_names}),
        )
        logits = outputs[0]
        pred = np.argmax(logits, axis=1)[0]
        return self.id2label[pred]

    def predict_with_html(self, text: str, path="./attention_avg.html") -> str:

        def generate_html(tokens, attention_weights, output_path="attention_avg.html"):
            norm_weights = (attention_weights - attention_weights.min()) / (
                attention_weights.max() - attention_weights.min() + 1e-8
            )
            html = "<html><head><style>span.token {padding:2px 5px;margin:2px;border-radius:5px;display:inline-block;font-family:monospace;}</style></head><body>"
            html += "<h2>平均 Attention 可视化 ([CLS] → tokens)</h2><div>"
            for token, weight in zip(tokens, norm_weights):
                red = int(255 * weight)
                color = f"rgba({red}, 0, 0, {weight:.2f})"
                html += f'<span class="token" style="background-color:{color}">{token}</span>'
            html += "</div></body></html>"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_mask = inputs["attention_mask"][0]
        seq_len = int(np.sum(attention_mask))

        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = cast(
            list[np.ndarray],
            self.session.run(output_names, {k: inputs[k] for k in input_names}),
        )

        # Extract logits and attentions
        logits, attentions = outputs
        pred = self.id2label[np.argmax(logits, axis=1)[0]]

        layer = 5  # 选择最后一层（MiniRBT 只有 6 层，索引0~5）

        all_heads = attentions[layer][0]  # shape: [num_heads, seq_len, seq_len]
        avg_attention = np.mean(all_heads, axis=0)  # shape: [seq_len, seq_len]
        cls_attention = avg_attention[0][:seq_len]
        tokens = tokens[:seq_len]

        # Generate HTML file
        generate_html(tokens, cls_attention, output_path="attention.html")
        return pred


if __name__ == "__main__":
    exit()
