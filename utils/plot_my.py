import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os


def plot_confusion_matrix(matrix, save_path):
    plt.clf()
    # 绘制矩阵
    fig, ax = plt.subplots(figsize=(18, 18))
    sns.set(font_scale=1.2)

    # 绘制矩阵
    sns.heatmap(matrix, annot=True, cmap="Greens", cbar=False, ax=ax, annot_kws={"size": 12})

    # 隐藏等于零的数字
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                ax.text(j + 0.5, i + 0.5, '', ha='center', va='center')

    ax.set_title('Noise labels matrix')
    ax.set_xlabel('Observed noise labels')
    ax.set_ylabel('Latent true labels')
    # 设置坐标轴不使用科学计数法
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.savefig(save_path)
    plt.close()


def plot_accuracy(epoch_list, train_acc_list, test_acc_list, save_path=None, train_label_name="tmp",
                  test_label_name="tmp"):
    """
    Plot accuracy against epoch.

    Args:
        epoch_list (list): List of epochs.
        accuracy_list (list): List of accuracies corresponding to each epoch.
    """
    plt.clf()
    plt.plot(epoch_list, train_acc_list, marker='o', linestyle='-', label=train_label_name)
    plt.plot(epoch_list, test_acc_list, marker='o', linestyle='-', label=test_label_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Accuracy vs. Epoch')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)  # 保存图片到指定路径
    plt.close()


def write_lists_to_file(list1, list2, file_path):  # 把两个list的内容写到txt文件，前者是下标，后者是内容
    with open(file_path, 'w') as f:
        for item1, item2 in zip(list1, list2):
            f.write(f"{item1} : {item2}\n")


def append_to_file(tensor, file_path, fmt='%d'):
    with open(file_path, 'a') as f:
        np.savetxt(f, tensor, fmt=fmt)


def write_matrix_to_file(matrix_np, file_path):  # 把二维numpy存储到txt文件中
    # 使用 numpy.savetxt() 函数将数组保存到文本文件
    np.savetxt(file_path, matrix_np, fmt='%d', delimiter='\t')


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


if __name__ == "__main__":
    # 创建随机数据集
    np.random.seed(42)  # 设置随机种子以获得可重复的结果
    epochs = list(np.arange(1, 11))  # 创建一个从1到10的周期列表
    train_accuracy = np.random.uniform(low=50.0, high=100.0, size=10)  # 生成50到100之间的随机训练准确率
    test_accuracy = np.random.uniform(low=50.0, high=100.0, size=10)  # 生成50到100之间的随机测试准确率

    # 调用函数绘制图表
    plot_accuracy(
        epoch_list=epochs,
        train_acc_list=train_accuracy,
        test_acc_list=test_accuracy,
        save_path='accuracy_plot.png',  # 指定保存图表的文件路径
        train_label_name='Training Accuracy',
        test_label_name='Test Accuracy'
    )
