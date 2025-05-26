import torch
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics_macro(output, target, average: str = 'macro'):
    """
    计算多分类的 Precision、Recall 和 F1 分数，使用 sklearn 内置函数。

    :param output: 模型输出的 logits 或概率值，shape=[batch_size, num_classes]
    :param target: 真实标签，shape=[batch_size]
    :param average: 聚合方式，可选 'macro', 'micro', 'weighted' 等
    :return: precision, recall, f1
    """
    # 1) 取最大 logit 对应的类作为预测
    _, predicted = torch.max(output, dim=1)

    # 2) 转到 CPU + numpy
    pred_np = predicted.cpu().numpy()
    tgt_np  = target   .cpu().numpy()

    # 3) 调用 sklearn，zero_division=0 防止分母为 0 抛错
    precision, recall, f1, _ = precision_recall_fscore_support(
        tgt_np, pred_np,
        average=average,
        zero_division=0
    )

    return precision, recall, f1

def calculate_metrics_micro(output, target, average: str = 'micro'):
    """
    计算多分类的 Precision、Recall 和 F1 分数，使用 sklearn 内置函数。

    :param output: 模型输出的 logits 或概率值，shape=[batch_size, num_classes]
    :param target: 真实标签，shape=[batch_size]
    :param average: 聚合方式，可选 'macro', 'micro', 'weighted' 等
    :return: precision, recall, f1
    """
    # 1) 取最大 logit 对应的类作为预测
    _, predicted = torch.max(output, dim=1)

    # 2) 转到 CPU + numpy
    pred_np = predicted.cpu().numpy()
    tgt_np  = target   .cpu().numpy()

    # 3) 调用 sklearn，zero_division=0 防止分母为 0 抛错
    precision, recall, f1, _ = precision_recall_fscore_support(
        tgt_np, pred_np,
        average=average,
        zero_division=0
    )

    return precision, recall, f1