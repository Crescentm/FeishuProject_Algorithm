import re
import pandas as pd
import csv



def clean_weibo_text(text):
    valid_keywords = [
        '笑', '哭', '泪', '心', '怒', '汗', '抱', '喜欢', '亲', '色', '偷笑', '害羞', '惊讶', '开心', '哭泣',
        '调皮', '害怕', '生气', '思考', '微笑', '呲牙', '委屈', '感动', '鼓掌', '加油', '抱拳', '拍手', '星星眼',
        '晕', '晚安', '睡觉', '吐', '呆萌', '抓狂', '拍砖', '爱', '尴尬', '大哭', '坏笑', '高兴', '害羞', '发怒',
        '兴奋', '酷', '赞', 'ok', '拜年', '卖萌', '抱抱', '转圈', '拜拜', '惊恐', '冷', '拜托', '拜谢', '炸裂',
        '流汗', '偷乐', '开心', '傻眼', '鄙视', '叹气', '纠结', '疑问', '点赞', '赞', '抱歉', '感恩', '感冒', '感情',
        '炸鸡', '雪人', '火', '狗', '猫', '熊', '兔', '猪', '骷髅', '鸡', '太阳', '月亮', '星', '花', '蛋糕',
        '巧克力', '糖果', '礼物', '礼花', '福', '平安', '红包', '祝福', '祝', '祝贺', '新年', '节日', '节', '圣诞',
        '生日', '万圣', '奥运', '火炬', '鼓掌', '加油', '胜利', '拥抱', '握手', '拳头', '挥手', '招手'
    ]
    
    if not isinstance(text, str):
        return ""

    # 去除 URL
    text = re.sub(r"http[s]?://\S+", "", text)

    # 去除转发链 //@用户:
    text = re.sub(r"//@\S+?:", "", text)
    # 去除正文中 @用户
    text = re.sub(r"@\S+", "", text)

    # 连续表情缩成一个
    text = re.sub(r"(\[[^\[\]]+\])\1+", r"\1", text)

    # 连续标点缩成一个
    text = re.sub(r"([！？!。，、,.，?])\1+", r"\1", text)

    # 利用 valid_keywords 过滤表情
    def filter_emoji(match):
        emoji = match.group(0)  # 带中括号的表情
        content = emoji[1:-1]   # 去掉中括号
        if any(keyword in content for keyword in valid_keywords):
            return emoji
        else:
            return ''

    text = re.sub(r"\[[^\[\]]+\]", filter_emoji, text)

    # 多空格替换成单空格，去首尾空格
    text = re.sub(r"\s+", " ", text).strip()

    # 去除无意义的词语
    text = text.replace("转发微博", "")

    return text

def process_csv(input_path, output_path):
    df = pd.read_csv(input_path, header=None, names=["text", "label"])
    df["text"] = df["text"].apply(clean_weibo_text)
    # 用 quoting=csv.QUOTE_ALL 确保每个字段都带引号，避免逗号换行破坏格式
    df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"清洗完成，结果保存至：{output_path}")

if __name__ == "__main__":
    input_file = "../data/base_train.csv"
    output_file = "../data/base_train_cleaned.csv"
    process_csv(input_file, output_file)
