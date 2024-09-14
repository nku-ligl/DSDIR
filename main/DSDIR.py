import os 
import sys

# import MADE.get_clean_epochs2 
# sys.path.append('..')

# 获取当前脚本所在的目录 (main/DS.py 的目录，即 main)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加 main 目录的父目录到系统路径，这样可以找到与 main 同级的 MADE 目录
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


import MADE

import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="根据命令行参数构造文件名。")

# 添加 noise_ratio 参数
parser.add_argument('--dataset_name', type=str, default='malicious_TLS-2023')

parser.add_argument('--noise_type', type=str, default='asym')

parser.add_argument('--noise_ratio', type=float, default=0.7)

# 添加 select_ratio 参数
parser.add_argument('--select_ratio', type=float, default=0.3, help='初筛比')

# 添加 beta 参数
parser.add_argument('--beta', type=float, default=-1, help='class-balanced loss参数, -1表示cross entropy, 0.9-0.999表示CB loss的参数')

# 添加 epoch 参数
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')

# 添加 epoch 参数
parser.add_argument('--warm_up', type=int, default=20, help='the number of warmup epochs')

# 添加 epoch 参数
parser.add_argument('--k', type=int, default=0.95, help='EMA参数')

parser.add_argument('--offset', type=float, default=0.5, help='偏移量')

parser.add_argument('--min_threshold', type=float, default=0.5, help='不确定性最小值')

# 解析命令行参数
args = parser.parse_args()


# 创建目录的函数
def ensure_dir(directory):
    # print('ok')
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def DS(model_dir, feat_dir, made_dir, result_dir, cuda, noise_ratio, epochs):
    
    # AE.train.main(data_dir, model_dir, cuda)
    # AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)
    # AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)
    # AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)
    TRAIN = 'be_ma_'
    TRAIN += str(round(noise_ratio,1))
    print(TRAIN, '??')
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20', noise_ratio, epochs, args.dataset_name, args.noise_type)
    if args.noise_type == 'asym':
        MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN, noise_ratio, args.dataset_name, args.noise_type)
    else:
        MADE.get_clean_epochs.main2(feat_dir, made_dir, '0.5', TRAIN, noise_ratio, args.dataset_name, args.noise_type)



import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from utils.MLP import *
from utils.processorbar import *
from utils.plot_my import *
from utils.metric import *
import argparse



net = None  # MLP
criterion = nn.CrossEntropyLoss()  # 计算loss
criterion2 = nn.CrossEntropyLoss(reduction='none')  # 这样才能返回每个样本的loss
optimizer = None
confusion_matrix = None  # 混淆矩阵
train_acc_list = []
test_acc_list = []
f = None
flag = None # 记录每个样本被标记为某个label的次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    def __init__(self, features, labels, true_labels):
        self.features = features
        self.labels = labels
        self.true_labels = true_labels
        self.indices = torch.arange(len(features))  # 保存数据的索引

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 返回数据、标签和索引
        return self.features[idx], self.labels[idx], self.true_labels[idx], self.indices[idx]


def save_loss_histograms(loss_dis, save_dir='result/loss_dic'):
    """
    绘制并保存每个类别的损失直方图。

    参数:
    - loss_dis: list of list，包含每个类别的损失值。
    - save_dir: str，保存图像的目录。
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每个类别的损失值列表
    for label, losses in enumerate(loss_dis):
        if len(losses) == 0:
            continue  # 如果类别的损失值列表为空，跳过该类别

        plt.hist(losses, bins=10, edgecolor='black')
        plt.title(f'Loss Distribution for Class {label}')
        plt.xlabel('Loss')
        plt.ylabel('Number of Samples')

        # 保存图像并关闭
        plt.savefig(os.path.join(save_dir, f'class_{label}_loss_distribution.png'))
        plt.close()


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta):
    """
    计算Class Balanced Loss的函数。

    参数：
    targets (Tensor): 真实标签的张量。
    logits (Tensor): 模型输出的张量。
    outputs (Tensor): 模型的预测输出张量。
    samples_per_cls (list): 每个类别的样本数量列表。
    no_of_classes (int): 类别的数量。
    loss_type (str): 损失函数的类型。
    beta (float): Class Balanced Loss中的beta参数。
    gamma (float): Class Balanced Loss中的gamma参数。

    返回：
    Tensor: 计算得到的Class Balanced Loss张量。
    """
    # 计算每个样本的权重
    # print('CB')
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.tensor(weights).float().to(logits.device)

    # 将标签转换为one-hot编码
    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    # 将权重转换为张量并扩展维度
    # weights = torch.tensor(weights).float().to(logits.device)
    weights = weights.clone().detach().float().to(logits.device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1)

    # 计算样本的加权和
    weights = weights * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)
    cb_loss = 0
    if loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


def train(net, epoch=100, train_loader=None, val_loader=None, beta=-1, samples_per_cls = None):  # 原始的训练
    print('\nEpoch: %d' % epoch)
    # print(len(cnt_per_class))
    # print(f.requires_grad)
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    for batch_idx, (inputs, targets, targets_true, idx) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 使用 scatter_add_ 来更新 cnt_per_class
        # if epoch == args.epochs:
            # cnt_per_class.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.int64))

        # print('target:', targets.shape)
        # print(inputs.shape, targets.shape)
        outputs = net(inputs)
        if args.beta != -1:
            loss = CB_loss(targets, outputs, samples_per_cls, len(samples_per_cls), 'softmax', beta)
        else:
            # print('outputs:', outputs)
            # print('targets:', targets)
            loss = criterion(outputs, targets)
        # print('loss:', loss)
        # print('epoch:', loss)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # if batch_idx < 5:

        # f[idx] = args.k * f[idx] + (1 - args.k) * _ # train的时候就维护f

        # 更新混淆矩阵
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    if epoch < args.warm_up:
        net.eval()
    # k = torch.tensor(args.k).to(device)
    for batch_idx, (inputs, targets, targets_true, idx) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)  # 对 outputs 应用 softmax
        max_values, predicted = outputs.max(1)
        idx_np = idx.numpy() # 转成numpy
        predicted_np = predicted.cpu().numpy()
        if epoch >= args.warm_up:
            # 1.loss >= threshold
            # 2.预测label == 标注label
            # print('1:', max_values.shape, f[idx].shape)
            # print('2:', predicted.shape, targets.shape)

            f_tmp = f[idx_np] + args.offset # 设置阈值，加上偏移
            f_tmp = torch.tensor(f_tmp, dtype=torch.float, requires_grad=False) # 转tensor，不需要梯度
            f_tmp = torch.clamp(f_tmp, max=0.99) # 将所有大于 0.99 的值设置为 0.99
            f_tmp = f_tmp.to(device) # 转cuda
            mask1 = max_values >= f_tmp # 阈值最多0.99，根据预测值初筛
            # weight = mask1 # 用于后续多指标筛选数据
    
            # 提取指定索引的数据
            flag_subset = flag[idx_np]

            # 计算每行的概率分布
            prob_dist = flag_subset / np.sum(flag_subset, axis=1, keepdims=True)

            # 计算每行的熵
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10), axis=1)  # 加上一个小常数防止log(0)

            # 归一化熵
            normalized_entropy = entropy / np.log(flag.shape[1])
            normalized_entropy = abs(normalized_entropy)
            normalized_entropy = torch.tensor(normalized_entropy, dtype=torch.float).to(device)  # 转为tensor


            threshold1 = 1 - f_tmp # 如果阈值算出来是0.01
            threshold1 = torch.clamp(threshold1, min=args.min_threshold) # 不设置最小值呢？

            threshold2 = 1 - f_tmp + args.offset

            

            # 比较归一化熵和阈值
            mask2 = normalized_entropy <= threshold1 # 小于等于阈值1
            # mask3 = (threshold1 < normalized_entropy) &  ( normalized_entropy <= threshold2 )  

            weight = mask1 & mask2 # 预测值高且不确定性很小
            # if epoch <= args.warm_up*2: # 热身二段，只用预测值筛
                # weight = mask1
            # weight = mask1
            if batch_idx < 5:
                # print('比较1:')
                # print(max_values)
                # print(f_tmp)
                # print('比较2:')
                # print(normalized_entropy)
                # print(threshold1)
                # 计算两个张量相等的元素数量
                equal_positions = torch.sum(mask1 == mask2)
                print('相等数量：', equal_positions)
                # 计算 mask1 为 True 且 mask2 为 False 的位置数量
                true_false_positions = torch.sum((mask1 == 1) & (mask2 == 0))

                # 计算 mask1 为 False 且 mask2 为 True 的位置数量
                false_true_positions = torch.sum((mask1 == 0) & (mask2 == 1))

                print("mask1单独的数量：", true_false_positions)
                print("mask2单独的数量：", false_true_positions)
                # print(mask2)
                # print(weight)

            # flag_subset_masked = flag_subset[mask3.cpu().numpy()]
            # if flag_subset_masked.size > 0:


            #     # 提取每个样本预测次数最高的两个标签及其权重
            #     top2_indices = np.argsort(flag_subset_masked, axis=1)[:, -2:]
            #     top2_values = np.take_along_axis(flag_subset_masked, top2_indices, axis=1)

            #     # 计算权重
            #     weights = top2_values / np.sum(top2_values, axis=1, keepdims=True)

            #     # 创建目标标签
            #     targets1 = top2_indices[:, 0]
            #     targets2 = top2_indices[:, 1]
            #     print(flag_subset)
            #     print(flag_subset_masked)
            #     print(top2_indices)
            #     print(top2_values)
            #     print(targets1)
            #     print(targets2)
            #     # 将目标标签和权重转换回张量
            #     targets1_tensor = torch.tensor(targets1, dtype=torch.long, device=device)
            #     targets2_tensor = torch.tensor(targets2, dtype=torch.long, device=device)
            #     weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

            #     # 计算每个标签对应的交叉熵损失
            #     loss1 = criterion(predicted[mask3], targets1_tensor)
            #     loss2 = criterion(predicted[mask3], targets2_tensor)
            # else:
                # 当 flag_subset_masked 为空时，跳过相应的操作
                # weights_tensor = torch.zeros((1, 2), dtype=torch.float32, device=device)  # 定义一个默认的权重
                # loss1 = torch.tensor(0.0, device=device)
                # loss2 = torch.tensor(0.0, device=device)

            # weighted_loss = weights_tensor[:, 0] * loss1 + weights_tensor[:, 1] * loss2


            targets[weight] = predicted[weight] # 用预测的值进行relabel
            loss = criterion(outputs[weight], targets[weight])  # 计算loss

            # if epoch == args.epochs:
            #     cnt_per_class.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.int64))


            # 计算权重
            # weight_size = weight.size(0)
            # mask3_size = mask3.size(0)
            # total_size = weight_size + mask3_size

            # weight_loss_ratio = weight_size / total_size
            # weighted_loss_ratio = mask3_size / total_size

            # 计算新的组合损失
            # combined_loss = weight_loss_ratio * loss + weighted_loss_ratio * weighted_loss

            # combined_loss = loss


            if weight.sum().item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            weight_np = weight.cpu().detach().numpy()
            flag[idx_np[weight_np], predicted[weight_np].cpu().detach().numpy()] += 1 # 哪些ID被标记为对应的label
        # elif epoch == args.warm_up-1:
            # flag[idx_np, predicted_np] += 1 # 热身阶段，全都记录，而不是只记录一部分
        max_values_np = max_values.cpu().detach().numpy()
        f[idx_np] = args.k * f[idx_np] + (1-args.k) * max_values_np

    # print('epochs:', epoch, cnt)
    train_acc_list.append(100. * correct / total)
    # print(TP+FP+TN+FN)
    torch.cuda.empty_cache()  # 清理缓存以释放内存

def test(net, epoch, test_loader, file_path):
    loss_dis = [[] for i in range(100)]  # loss分布
    confusion_matrix[:] = 0  # 混淆矩阵清空
    print('\nEpoch: %d' % epoch)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, targets_true, idx) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # 1.测试集的数据都以为是良性数据0，但恶意数据并不是0.
            # 2.我们真正拿到这个数据集的时候，loss都是按label 0去算的，根据这个loss看分布。（相当于先验已经知道了每个样本的真实label，这样才能算分布）

            # criterion2 = nn.CrossEntropyLoss(reduction='none')  # 这样才能返回每个样本的loss
            # targets2 = torch.zeros_like(targets, dtype=torch.long)  # 构造一个全良性（0）的数据
            # loss2 = criterion2(outputs, targets2)
            #
            # for i, target in enumerate(targets.cpu().numpy()):
            #     loss_dis[target.item()].append(loss2[i].item())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新混淆矩阵
            increment = torch.ones(targets.size())
            update_confusion_matrix(confusion_matrix, targets, predicted, increment)

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        test_acc_list.append(100. * correct / total)
    # print(confusion_matrix)
    if epoch == 1:
        write_matrix_to_file(confusion_matrix, file_path)
    else:
        append_to_file(confusion_matrix, file_path)
    
def DIR():

    global f,flag,confusion_matrix

    # 根据参数构造文件名
    be_ceshi_filename = f'data/feat/{args.dataset_name}/{args.noise_type}/be_ceshi_{args.noise_ratio}_{args.select_ratio}.npy'
    ma_ceshi_filename = f'data/feat/{args.dataset_name}/{args.noise_type}/ma_ceshi_{args.noise_ratio}_{args.select_ratio}.npy'
    ceshi_filename = f'data/feat/{args.dataset_name}/{args.noise_type}/ceshi_{args.noise_ratio}_{args.select_ratio}.npy'
    remaining_file = f'data/feat/{args.dataset_name}/{args.noise_type}/remaining_0_ceshi_{args.noise_ratio}_{args.select_ratio}.npy'

    # 使用构造的文件名加载数据
    be_ceshi = np.load(be_ceshi_filename) # MADE筛选的良性数据
    ma_ceshi = np.load(ma_ceshi_filename) # 拿到手里的恶意数据，asym保证他们的干净性
    ceshi = np.load(ceshi_filename) # 测试集
    val_ceshi = np.load(remaining_file) #验证集，其实是MADE筛完以后不确定的数据。

    # 假设数据格式：features 在前，labels 在最后一列
    # 假设数据格式：features 在前，labels 在最后一列
    X_train = np.vstack((be_ceshi[:, :-2], ma_ceshi[:, :-2]))
    y_train = np.hstack((be_ceshi[:, -2], ma_ceshi[:, -2]))
    y_train_true = np.hstack((be_ceshi[:, -1], ma_ceshi[:, -1]))


    X_test = ceshi[:, :-2]
    y_test = ceshi[:, -2]
    y_test_true = ceshi[:, -1]


    X_val = val_ceshi[:, :-2]
    y_val = val_ceshi[:, -2]
    y_val_true = val_ceshi[:, -1]

    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_val:', X_val.shape)
    print('y_val:', y_val.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    # 计算每个标签的出现次数
    unique_labels, counts = np.unique(y_val, return_counts=True)
    # 遍历 unique_labels 和 counts
    print('验证集:')
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label} 出现了 {count} 次")


    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_train_true_tensor = torch.tensor(y_train_true, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    y_test_true_tensor = torch.tensor(y_test_true, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_val_true_tensor = torch.tensor(y_val_true, dtype=torch.long)


    # 创建 PyTorch 数据集和数据加载器
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor, y_train_true_tensor)
    val_dataset = CustomDataset(X_val_tensor, y_val_tensor, y_val_true_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor, y_test_true_tensor)


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 获取数据特征的维度和标签的数量
    input_dim = X_train.shape[1]
    unique_y_train = np.unique(y_train)
    unique_y_val = np.unique(y_val)
    unique_y_test = np.unique(y_test)
    # exit(0)
    output_dim = len(np.unique(np.concatenate((unique_y_train, unique_y_test, unique_y_val))))

    cnt_per_class = torch.zeros(output_dim, dtype=torch.int64, device=device)


    f = np.zeros(X_val.shape[0])
    flag = np.zeros((X_val.shape[0], output_dim))

    # 计算每个标签的出现次数
    unique_labels, counts = np.unique(y_train, return_counts=True)

    print('训练集:')
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label} 出现了 {count} 次")

    samples_per_cls = [1] * output_dim  # 存储每个类别的样本数量
    # print(samples_per_cls)
    # 将结果打印出来
    for i in range(len(counts)):
        samples_per_cls[int(unique_labels[i])] = counts[i]

    print(samples_per_cls)

    # print('input:', input_dim, 'output:', output_dim)

    confusion_matrix = torch.zeros(output_dim, output_dim)  # m*m的全0混淆矩阵

    # beta = 0.0 # beta设置

    # 初始化 MLP
    net = MLP(num_features=input_dim, num_labels=output_dim)  # 网络

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    global optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                        momentum=0.9)

    print('输入维度和输出维度:', input_dim, output_dim)

    # 打印数据集的长度
    print(f"训练集的长度: {len(train_loader.dataset)}")
    print(f"测试集的长度: {len(test_loader.dataset)}")

    print('beta:', args.beta)
    epochs = range(1, 1 + args.epochs)  # epochs
    epochs = list(epochs)


    # 构建文件名
    file_name = f'result/{args.dataset_name}/{args.noise_type}/ratio_{args.noise_ratio}/select_{args.select_ratio}/beta_{args.beta}/warmup_{args.warm_up}'

    # 构建完整的文件路径
    txt_confu_name = f'{file_name}/confu.txt'
    txt_confu_analyze_name = f'{file_name}/confu_{args.min_threshold}_analyze.txt'
    figure_acc_name = f'{file_name}/acc.png'
    figure_confu_name = f'{file_name}/confu.png'

    # 确保目录存在
    os.makedirs(os.path.dirname(txt_confu_name), exist_ok=True)
    os.makedirs(os.path.dirname(txt_confu_analyze_name), exist_ok=True)
    os.makedirs(os.path.dirname(figure_acc_name), exist_ok=True)
    os.makedirs(os.path.dirname(figure_confu_name), exist_ok=True)

    print(type(net))

    for epoch in epochs:
        train(net, epoch, train_loader, val_loader, args.beta, samples_per_cls)
        test(net, epoch, test_loader, txt_confu_name)

    from collections import Counter

    # 初始化计数器
    target_counter = Counter()
    correct_counter = Counter()

    tot = 0
    # with open('targets_labels.txt', 'w') as file:
    for batch_idx, (inputs, targets, targets_true, idx) in enumerate(val_loader):
        predicted_labels = np.argmax(flag[idx], axis=1)
        targets = targets.cpu().detach().numpy()
        # if batch_idx < 5:
        #     print(targets.shape)
        #     print(predicted_labels.shape)
        #     print(targets == predicted_labels)
        eq_idx = targets == predicted_labels
        num = eq_idx.sum()

        # 统计 targets 中每个数的出现次数
        target_counter.update(targets)

        # 统计 eq_idx 为 True 时，targets 中每个数的出现次数
        correct_counter.update(targets[eq_idx])


        tot += num

    print(tot, X_val.shape[0], tot/X_val.shape[0])

    # 输出结果
    print("Targets中每个数的出现次数:", dict(target_counter))
    print("Targets对应eq_idx下标中每个数的出现次数:", dict(correct_counter))

    # 计算并输出每个数在 targets 中被正确预测的比例
    accuracy_per_class = {}
    for key in target_counter:
        accuracy_per_class[key] = correct_counter[key] / target_counter[key] if target_counter[key] > 0 else 0

    print("每个数被正确预测的比例:", accuracy_per_class)

    plot_accuracy(
        epoch_list=epochs,
        train_acc_list=train_acc_list,
        test_acc_list=test_acc_list,
        save_path=figure_acc_name,  # 指定保存图表的文件路径
        train_label_name='Training Accuracy',
        test_label_name='Test Accuracy'
    )

    plot_confusion_matrix(confusion_matrix, figure_confu_name)

    sorted_results, categories = analyze_confusion_matrix(confusion_matrix)

    save_results_to_file(txt_confu_analyze_name, sorted_results, categories)


def main():
    
    feat_dir = 'data/feat/' + args.dataset_name + '/'  + args.noise_type
    model_dir= 'data/model'
    made_dir = 'data/made'
    result_dir='data/result'

    # 确保目录存在
    ensure_dir(feat_dir)
    ensure_dir(model_dir)
    ensure_dir(made_dir)
    ensure_dir(result_dir)

    cuda = 0
    DS(model_dir, feat_dir, made_dir, result_dir, cuda, args.noise_ratio, args.epochs)
    DIR()


if __name__ == '__main__':
    main()