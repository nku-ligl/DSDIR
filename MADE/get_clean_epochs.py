from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import pandas as pd
import altair as alt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def plot_sample_density(be_extract, be_extract_NlogP, save_path='sample_density_by_label_.png'):
    """
    绘制样本密度图并保存为文件。

    参数:
    - be_extract: 包含样本及其标签的数组，be_extract[i][-1] 表示第 i 个样本的标签。
    - be_extract_NlogP: 每个样本的样本密度。
    - save_path: 保存图形的路径（默认 'sample_density_by_label_.png'）。
    """
    # 提取标签和样本密度
    labels = [sample[-1] for sample in be_extract]
    densities = be_extract_NlogP

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制每个点
    for i in range(len(labels)):
        color = 'blue' if labels[i] == 0 else 'red'
        plt.scatter(labels[i], densities[i], color=color, alpha=0.6, edgecolors='w', s=100)

    # 设置横轴刻度为 0 和 1
    plt.xticks([0, 1], ['0', '1'])

    # 添加图例和标签
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Label 0',
                            markerfacecolor='blue', markersize=10, alpha=0.6)
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Label 1',
                           markerfacecolor='red', markersize=10, alpha=0.6)
    plt.legend(handles=[blue_patch, red_patch])
    plt.title('Sample Density by Label')
    plt.xlabel('Label')
    plt.ylabel('Sample Density')

    # 保存图形
    plt.savefig(save_path)

    # 关闭图形
    plt.close()


def main(feat_dir, made_dir, alpha, TRAIN, corruption_ratio, data_name, corruption_type):
    # According to MADE-density, select true benign samples
    alpha = float(alpha)
    be = np.load(os.path.join(feat_dir, 'be_' + str(round(corruption_ratio, 1)) + '.npy'))
    ma = np.load(os.path.join(feat_dir, 'ma_' + str(round(corruption_ratio, 1)) + '.npy'))
    test = np.load(os.path.join(feat_dir, 'test_' + str(round(corruption_ratio, 1)) + '.npy'))
    feats = np.concatenate((be, ma), axis=0)
    print(feats.shape, '???')

    # 获取最后一列的标签
    labels = feats[:, -1]

    # 找出所有不同的标签和它们的数量
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 打印每个标签和对应的数量
    for label, count in zip(unique_labels, counts):
        print(f"Label: {label}, Count: {count}")
    # exit(0)

    be_number, be_shape = be.shape
    ma_number, ma_shape = ma.shape
    assert (be_shape == ma_shape)
    NLogP = [0 for _ in range(be_number + ma_number)]
    nlogp_lst = [[] for _ in range(be_number + ma_number)]

    epochs = 0
    for filename in os.listdir(made_dir):
        if filename.startswith("be"):
            epochs += 1
    print(epochs, '???')
    # 之前的代码，其实不需要匹配这么精确，除非AE阶段产生了一些其他文件


    # 获取所有文件列表
    files = os.listdir(made_dir)
    # 处理所有以 be 或 ma 开头的文件
    
    for filename in files:
        # if filename.startswith('be_') or filename.startswith('ma_'):
        if data_name in filename and TRAIN in filename and corruption_type in filename: # 得是对应数据集的、be_ma的、对应噪声类型的，不然把之前的结果也用上了
            # 修改正则表达式以匹配数字后跟着的 _TRAIN
            
            # 直接将 TRAIN 变量的值插入到正则表达式中
            # match = re.match(r'^(be_|ma_)(\d+\.\d+)(_{' + TRAIN + '})?', filename)

            match = re.match(r'^(be_|ma_)(\d+\.\d+)', filename)


            # if match:
            x_value = float(match.group(2))  # 提取并转换为浮点数
            if abs(x_value - corruption_ratio) <= 0.01:
                # print(TRAIN, ':', x_value, corruption_ratio)
                is_be_file = filename.startswith('be_')

                with open(os.path.join(made_dir, filename), 'r') as fp:
                    print(be_number, ma_number, ':', be_number + ma_number)
                    cnt = 0
                    for i, line in enumerate(fp):
                        cnt += 1
                        # if i + be_number > be_number + ma_number:
                            # print(i, '???')

                        s = float(line.strip())
                        if s > 10000:
                            s = 10000

                        if is_be_file:
                            NLogP[i] = NLogP[i] + s
                            nlogp_lst[i].append(s)
                        else:
                            NLogP[i + be_number] = NLogP[i + be_number] + s
                            nlogp_lst[i + be_number].append(s)
                    print(filename, ':', cnt)

    path_prefix = f'data/result_DS/{data_name}/{corruption_type}/ratio_{corruption_ratio}/'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)


    file_path = f"NLogP_{data_name}_{str(corruption_ratio)}.txt"

    with open(path_prefix + file_path, 'w') as f:
        print('open success:', path_prefix + file_path)
        for item in NLogP:
            f.write(f"{item}\n")

    seq = list(range(len(NLogP)))
    seq.sort(key=lambda x: NLogP[x], reverse=True)

    be_extract = []
    be_extract_tsne = []  # 用来维护t-sne
    density_idx = []  # 维护前一半的index
    be_extract_lossline = []
    be_extract_NLogP = []
    # extract_range = int(alpha * (be_number + ma_number)) # 选一半
    extract_range = min(len(feats), int(be_number + ma_number))
    for i in range(extract_range):
        if i < extract_range // 2:
            be_extract_tsne.append(feats[seq[i]])
            density_idx.append(seq[i])
        be_extract.append(feats[seq[i]])
        be_extract_lossline.append(nlogp_lst[seq[i]])
        be_extract_NLogP.append(NLogP[seq[i]])

    density_idx = np.array(density_idx)
    density_idx = np.sort(density_idx)

    length = len(seq)
    percentages = [i / 10 for i in range(1, 11)]  # 生成 0.1 到 1.0 的百分比列表
    checkpoints = [int(p * length) for p in percentages]
    counts = {cp: 0 for cp in checkpoints}

    cnt = 0
    with open(path_prefix + f'log_label_{data_name}_{str(corruption_ratio)}.txt', 'w') as file:
        # 追加内容到文件
        for i in range(len(seq)):
            file.write(f"{i}: {NLogP[seq[i]]}, label: {feats[seq[i]][-1]}\n")
            # if feats[seq[i]][-1] > 14:
            #     print(seq[i], ':', feats[seq[i]])
    for i in range(len(seq)):
        # print(i, ':', NLogP[seq[i]], ', label:', feats[seq[i]][-1])
        feat = feats[seq[i]]
        if feat[-1] == 0:  # 良性
            cnt += 1
        if i + 1 in checkpoints:
            counts[i + 1] = cnt

    with open(path_prefix + f'output_{data_name}_{str(corruption_ratio)}.txt', 'a') as file:
        for cp in checkpoints:
            percentage = (counts[cp] / cp) * 100
            # 格式化字符串并写入文件
            file.write(f"At {int(cp / length * 100)}%: 良性 = {counts[cp]}, 全部 = {cp}, 比例 = {percentage:.2f}%\n")
            print(f"At {int(cp / length * 100)}%: 良性 = {counts[cp]}, 全部 = {cp}, 比例 = {percentage:.2f}%")

    # 初始化字典来存储总损失和样本数
    label_loss_sum = defaultdict(float)
    label_count = defaultdict(int)

    # 遍历所有样本
    for i in range(len(feats)):
        label = feats[i][-1]
        loss = NLogP[i]
        label_loss_sum[label] += loss
        label_count[label] += 1

    # 计算每个标签的平均损失
    label_avg_loss = {label: label_loss_sum[label] / label_count[label] for label in label_loss_sum}

    # 输出每个标签的平均损失
    for label, avg_loss in label_avg_loss.items():
        # print(label, label_loss_sum[label], label_count[label])
        print(f"Label {label}: Average Loss = {avg_loss}")
        print(f"Label {label}: Total Number = {label_count[label]}")

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # 示例数据
    # feats = ...  # 你的 feats 数据

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ratio_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    def add_last_column(arr):
        return np.column_stack((arr, arr[:,-1]))
    
    from collections import Counter

    # 统计原始数据最后一列标签及其数量
    original_last_column = ma[:, -1]
    original_label_counts = Counter(original_last_column)

    print("执行前的最后一列标签数量:")
    for label, count in original_label_counts.items():
        print(f"标签 {label}: {count} 个")

    ma = add_last_column(ma)

    # 统计添加最后一列后的数据最后一列标签及其数量
    new_last_column = ma[:, -1]
    new_label_counts = Counter(new_last_column)

    print("执行后的最后一列标签数量:")
    for label, count in new_label_counts.items():
        print(f"标签 {label}: {count} 个")


    test = add_last_column(test)

    for ratio in ratio_list:
        # 计算平均值
        tmp_features = np.array(be_extract[:int(ratio * len(be_extract))])
        le_features = np.array(be_extract[int(ratio * len(be_extract)):])
        
        tmp_features = add_last_column(tmp_features)
        le_features = add_last_column(le_features)

        # 1.存储良性样本
        # 将 tmp_features 的最后一列的值都改成 0, 因为筛选的样本都以为是良性，不能用真实的label
        tmp_features[:, -2] = 0
        
        # 没筛选的真实标签
        np.save(os.path.join(feat_dir, 'remaining_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(
            round(ratio, 1)) + '.npy'), le_features)
        le_features[:, -2] = 0
        np.save(
            os.path.join(feat_dir, 'be_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            tmp_features)  # 第一个数字是噪声率，第二个数字是初筛前p%
        # 2.存储恶意样本
        np.save(
            os.path.join(feat_dir, 'ma_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            ma)  # 注意这里ma_ceshi的内容是固定的，因为噪声率才是和ma_ceshi对应的，初筛比例变动对ma、test是没有影响的，
        # 3.存储没筛选的样本
        np.save(os.path.join(feat_dir, 'remaining_0_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(
            round(ratio, 1)) + '.npy'), le_features)
        # 4.存储测试集
        np.save(
            os.path.join(feat_dir, 'ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            test)

    # tmp_features = tmp_features[:, :-1]
    # mean_features = np.mean(tmp_features, axis=0)

    be_extract_np = np.array(be_extract)
    features_all = be_extract_np[:, :-1]
    labels_all = be_extract_np[:, -1]

    return

def main2(feat_dir, made_dir, alpha, TRAIN, corruption_ratio, data_name , corruption_type):
    # According to MADE-density, select true benign samples
    alpha = float(alpha)
    be = np.load(os.path.join(feat_dir, 'be_' + str(round(corruption_ratio, 1)) + '.npy'))
    ma = np.load(os.path.join(feat_dir, 'ma_' + str(round(corruption_ratio, 1)) + '.npy'))
    test = np.load(os.path.join(feat_dir, 'test_' + str(round(corruption_ratio, 1)) + '.npy'))
    feats = np.concatenate((be, ma), axis=0)
    print(feats.shape, '???')
    # print(feats[:100,-1], 'look')

    # 使用 floor 确保值向下取整为整数
    # feats[:, -1] = np.floor(feats[:, -1]).astype(int)

    labels = feats[:, -1].astype(int)

    # print(feats[:, -1].dtype, '!!!')
    print(labels[-1].dtype, '!!!!!')

    # 找出所有不同的标签和它们的数量
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 打印每个标签和对应的数量
    for label, count in zip(unique_labels, counts):
        print(f"Label: {label}, Count: {count}")
    # exit(0)

    be_number, be_shape = be.shape
    ma_number, ma_shape = ma.shape
    assert (be_shape == ma_shape)
    NLogP = [0 for _ in range(be_number + ma_number)]
    nlogp_lst = [[] for _ in range(be_number + ma_number)]

    epochs = 0
    for filename in os.listdir(made_dir):
        if filename.startswith("be"):
            epochs += 1
    print(epochs, '???')
    # 之前的代码，其实不需要匹配这么精确，除非AE阶段产生了一些其他文件


    # 获取所有文件列表
    files = os.listdir(made_dir)
    # 处理所有以 be 或 ma 开头的文件
    
    for filename in files:
        # if filename.startswith('be_') or filename.startswith('ma_'):
        if data_name in filename:
            # 使用正则表达式匹配 "be_" 或 "ma_" 开头，后面跟一个小数
            match = re.match(r'^(be_|ma_)(\d+\.\d+)', filename)

            if match:
                x_value = float(match.group(2))  # 提取并转换为浮点数
                if abs(x_value - corruption_ratio) <= 0.01:
                    is_be_file = filename.startswith('be_')

                    with open(os.path.join(made_dir, filename), 'r') as fp:
                        for i, line in enumerate(fp):
                            s = float(line.strip())
                            if s > 10000:
                                s = 10000

                            if is_be_file:
                                NLogP[i] = NLogP[i] + s
                                nlogp_lst[i].append(s)
                                # print(i, ':', NLogP[i])
                            else:
                                NLogP[i + be_number] = NLogP[i + be_number] + s
                                nlogp_lst[i + be_number].append(s)



    file_path = f"NLogP_{data_name}_{str(corruption_ratio)}.txt"
    
    path_prefix = f'data/result_DS/{data_name}/{corruption_type}/ratio_{corruption_ratio}/'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    with open(path_prefix + file_path, 'w') as f:
        for item in NLogP:
            f.write(f"{item}\n")

    seq = list(range(len(NLogP)))
    seq.sort(key=lambda x: NLogP[x], reverse=True)

    be_extract = []
    be_extract_tsne = []  # 用来维护t-sne
    density_idx = []  # 维护前一半的index
    be_extract_lossline = []
    be_extract_NLogP = []

    extract_range = min(len(feats), int(be_number + ma_number))
    num_classes = len(set(labels))  # 假设labels是所有样本的类别标签列表

    for i in range(num_classes):
        be_extract.append([])  # 初始化每个类的列表
        be_extract_tsne.append([])  # 初始化每个类的t-SNE列表
        be_extract_lossline.append([])  # 初始化每个类的lossline列表
        be_extract_NLogP.append([])  # 初始化每个类的NLogP列表

    for i in range(extract_range):
        class_index = labels[seq[i]]  # 获取当前样本的类别索引
        if class_index < extract_range // 2:
            be_extract_tsne[class_index].append(feats[seq[i]])
            density_idx.append(seq[i])
        be_extract[class_index].append(feats[seq[i]])
        be_extract_lossline[class_index].append(nlogp_lst[seq[i]])
        be_extract_NLogP[class_index].append(NLogP[seq[i]])

    # density_idx = np.array(density_idx)
    # density_idx = np.sort(density_idx)

    # length = len(seq)
    # percentages = [i / 10 for i in range(1, 11)]  # 生成 0.1 到 1.0 的百分比列表
    # checkpoints = [int(p * length) for p in percentages]
    # counts = {cp: 0 for cp in checkpoints}

    # cnt = 0
    # with open(f'log_label_{data_name}_{str(corruption_ratio)}.txt', 'w') as file:
    #     # 追加内容到文件
    #     for i in range(len(seq)):
    #         file.write(f"{i}: {NLogP[seq[i]]}, label: {feats[seq[i]][-1]}\n")
    #         # if feats[seq[i]][-1] > 14:
    #         #     print(seq[i], ':', feats[seq[i]])
    # for i in range(len(seq)):
    #     # print(i, ':', NLogP[seq[i]], ', label:', feats[seq[i]][-1])
    #     feat = feats[seq[i]]
    #     if feat[-1] == 0:  # 良性
    #         cnt += 1
    #     if i + 1 in checkpoints:
    #         counts[i + 1] = cnt

    # with open(f'output_{data_name}_{str(corruption_ratio)}.txt', 'a') as file:
    #     for cp in checkpoints:
    #         percentage = (counts[cp] / cp) * 100
    #         # 格式化字符串并写入文件
    #         file.write(f"At {int(cp / length * 100)}%: 良性 = {counts[cp]}, 全部 = {cp}, 比例 = {percentage:.2f}%\n")
    #         print(f"At {int(cp / length * 100)}%: 良性 = {counts[cp]}, 全部 = {cp}, 比例 = {percentage:.2f}%")

    # 初始化字典来存储总损失和样本数
    label_loss_sum = defaultdict(float)
    label_count = defaultdict(int)

    # 遍历所有样本
    for i in range(len(feats)):
        label = feats[i][-1]
        loss = NLogP[i]
        label_loss_sum[label] += loss
        label_count[label] += 1

    # 计算每个标签的平均损失
    label_avg_loss = {label: label_loss_sum[label] / label_count[label] for label in label_loss_sum}

    # 输出每个标签的平均损失
    for label, avg_loss in label_avg_loss.items():
        # print(label, label_loss_sum[label], label_count[label])
        print(f"Label {label}: Average Loss = {avg_loss}")
        print(f"Label {label}: Total Number = {label_count[label]}")

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # 示例数据
    # feats = ...  # 你的 feats 数据

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ratio_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    def add_last_column(arr):
        return np.column_stack((arr, arr[:,-1]))
    
    from collections import Counter

    # 统计原始数据最后一列标签及其数量
    original_last_column = ma[:, -1]
    original_label_counts = Counter(original_last_column)

    print("执行前的最后一列标签数量:")
    for label, count in original_label_counts.items():
        print(f"标签 {label}: {count} 个")

    ma = add_last_column(ma)

    # 统计添加最后一列后的数据最后一列标签及其数量
    new_last_column = ma[:, -1]
    new_label_counts = Counter(new_last_column)

    print("执行后的最后一列标签数量:")
    for label, count in new_label_counts.items():
        print(f"标签 {label}: {count} 个")


    test = add_last_column(test)

    for ratio in ratio_list:
        all_tmp_features = []
        all_le_features = []

        for class_index in range(len(be_extract)):  # 遍历每个类别
            class_data = be_extract[class_index]  # 获取当前类别的数据
            
            # 计算 tmp_features 和 le_features 的分割点
            split_index = int(ratio * len(class_data))
            
            # 分割数据
            tmp_features = np.array(class_data[:split_index])
            le_features = np.array(class_data[split_index:])
            
            # 添加最后一列
            tmp_features = add_last_column(tmp_features)
            le_features = add_last_column(le_features)
            
            # 将当前类别的数据添加到汇总列表中
            all_tmp_features.extend(tmp_features)
            all_le_features.extend(le_features)

            

        # 转换为 numpy 数组
        all_tmp_features = np.array(all_tmp_features)
        all_le_features = np.array(all_le_features)

        print(ratio, ':', len(all_tmp_features), len(all_le_features) )

        # 存储汇总后的良性样本和未筛选样本
        np.save(
            os.path.join(feat_dir, 'be_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            all_tmp_features)  # 第一个数字是噪声率，第二个数字是初筛前p%
        np.save(
            os.path.join(feat_dir, 'ma_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            ma)  # 存储恶意样本
        np.save(
            os.path.join(feat_dir, 'remaining_0_ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            all_le_features)  # 存储没筛选的样本
        np.save(
            os.path.join(feat_dir, 'ceshi_' + str(round(corruption_ratio, 1)) + '_' + str(round(ratio, 1)) + '.npy'),
            test)  # 存储测试集
