import torch
import numpy as np


def compute_metrics(targets, predicted, idx):
    """
    计算指定类别的TP、FP、TN、FN
    :param targets: 真实标签
    :param predicted: 预测标签
    :param idx: 要计算的正类样本类别索引
    :return: (TP, FP, TN, FN)
    """
    # 初始化计数器
    TP, FP, TN, FN = 0, 0, 0, 0

    # 使用张量操作计算指标
    TP = torch.sum((targets == idx) & (predicted == idx)).item()
    FP = torch.sum((targets != idx) & (predicted == idx)).item()
    TN = torch.sum((targets != idx) & (predicted != idx)).item()
    FN = torch.sum((targets == idx) & (predicted != idx)).item()

    return TP, FP, TN, FN


# 更新混淆矩阵
def update_confusion_matrix(matrix, tensor1, tensor2, increment):  # 在所有(tensor1,tensor2)的下标处对matrix += 1

    # 将 matrix 在指定位置累加
    matrix.index_put_((tensor1, tensor2), increment, accumulate=True)


# 根据
def calculate_precision_recall(confusion_matrix):
    # 计算每个类别的精确度和召回率
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP

        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


def analyze_confusion_matrix(confusion_matrix):
    # 样本数量
    sample_counts = confusion_matrix.sum(axis=1)

    # 准确率（Accuracy）
    accuracy_per_class = np.diag(confusion_matrix) / sample_counts * 100

    # 召回率（Recall）
    recall_per_class = np.diag(confusion_matrix) / sample_counts * 100

    # 将结果存储在一个字典中
    results = {
        "Class": np.arange(len(sample_counts)),
        "Sample Count": sample_counts,
        "Accuracy": accuracy_per_class,
        "Recall": recall_per_class
    }

    # 转换为结构化数组以便排序
    dtype = [('Class', int), ('Sample Count', int), ('Accuracy', float), ('Recall', float)]
    structured_results = np.array([tuple(results[key][i] for key in results) for i in range(len(sample_counts))],
                                  dtype=dtype)

    # 按样本数量排序
    sorted_results = np.sort(structured_results, order='Sample Count')

    # 按20%、30%、50%比例划分
    n = len(sample_counts)
    many_threshold = int(n * 0.2)
    medium_threshold = int(n * 0.5)

    categories = np.empty(n, dtype='<U6')  # 字符串数组用于存储分类结果
    categories[:many_threshold] = 'many'
    categories[many_threshold:medium_threshold] = 'medium'
    categories[medium_threshold:] = 'few'

    return sorted_results, categories


# 分别计算many, medium, few的平均accuracy并输出详细信息
def save_results_to_file(file_path, sorted_results, categories):
    def category_info(category_name):
        indices = np.where(categories == category_name)[0]
        category_results = sorted_results[indices]
        average_accuracy = np.mean(category_results['Accuracy'])
        result = [f"{category_name}, average accuracy: {average_accuracy:.2f}%"]
        for res in category_results:
            result.append(
                f"Class: {res['Class']}, Sample Count: {res['Sample Count']}, Accuracy: {res['Accuracy']:.2f}%, Recall: {res['Recall']:.2f}%")
        return "\n".join(result)

    with open(file_path, 'w') as f:
        f.write(category_info('many'))
        f.write("\n\n")
        f.write(category_info('medium'))
        f.write("\n\n")
        f.write(category_info('few'))





# 假设的标签数据和预测数据
targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 2, 1, 0])
predicted = torch.tensor([0, 2, 2, 0, 1, 1, 0, 2, 2, 1])

# 假设正类样本的索引
idx = 0

# 计算并输出结果
TP, FP, TN, FN = compute_metrics(targets, predicted, idx)

print(TP)
print(FP)
print(TN)
print(FN)

if __name__ == "__main__":
    # 示例混淆矩阵
    confusion_matrix = np.array([
        [589, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 599, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 315, 141, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 0, 0, 0],
        [0, 0, 72, 363, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0],
        [0, 0, 0, 0, 589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0],
        [21, 0, 1, 2, 0, 519, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 593, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
        [6, 0, 0, 0, 0, 15, 0, 569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 64, 382, 0, 0, 0, 0, 0, 0, 0, 0, 0, 154, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 561, 0, 0, 0, 0, 0, 2, 0, 0, 37, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 591, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 569, 0, 0, 0, 0, 0, 31, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 565, 14, 0, 0, 0, 15, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2, 4, 559, 1, 1, 0, 27, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 2, 1, 2, 37, 432, 0, 0, 119, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 590, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 591, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0],
        [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 590]
    ])

    # 使用函数分析混淆矩阵，划分many、medium、few
    sorted_results, categories = analyze_confusion_matrix(confusion_matrix)

    file_path = 'test.txt'
    save_results_to_file(file_path, sorted_results, categories)
