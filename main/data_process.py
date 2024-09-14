import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="根据命令行参数构造文件名。")

# 添加 corruption_ratio 参数
parser.add_argument('--dataset_name', type=str, default='BoAu')

# 添加 corruption_ratio 参数
parser.add_argument('--noise_type', type=str, default='asym')



# 解析命令行参数
args = parser.parse_args()

class CICIDS2017Preprocessor(object):
    def __init__(self, data_name, data_path, training_size, validation_size, testing_size):
        self.data_name = data_name
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size

        self.data = None
        self.features = None
        self.labels = None
        self.labels_encoded = []  # labels_encoded[i]表示数字i对应的label是什么

    def read_data(self):
        """"""
        filenames = glob.glob(os.path.join(self.data_path, '*.csv'))
        print(self.data_path)
        print(filenames)
        datasets = [pd.read_csv(filename, encoding='latin1', low_memory=True) for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        # self.data.drop(labels=['fwd_header_length.1'], axis=1, inplace=True)

    def _clean_column_name(self, column):
        """"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

    def remove_duplicate_values(self):
        """"""
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        """"""
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        """"""
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, how='any', inplace=True)

    def find_non_numeric_columns(self):  # 删除非数值列
        label_columns = ['label', 'label_category']
        self.data = self.data.drop(columns=label_columns)
        # print(self.data.head(1))
        # non_numeric_columns = self.data.select_dtypes(exclude=['number']).columns  # 找到非数值列
        # self.non_numeric_data = self.data[non_numeric_columns]  # 存储非数值列
        # self.data.drop(columns=non_numeric_columns, inplace=True)  # 删除非数值列
        # return non_numeric_columns

    def remove_constant_features(self, threshold=0.01):
        """"""
        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.items() if std < threshold]

        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features_pre(self, threshold=0.98):
        """"""
        # Correlation matrix
        data_corr = self.data.corr()

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.9):
        """
        Remove highly correlated features from the data.
        """
        # Select only the numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])

        # Correlation matrix
        data_corr = numeric_data.corr()
        # print(data_corr)
        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """"""
        # Proposed Groupings
        attack_group = None
        # if self.data_name == 'CIC-IDS-2017':
        #     attack_group = {
        #         'BENIGN': 'Benign',
        #         'PortScan': 'PortScan',
        #         'DDoS': 'DoS/DDoS',
        #         'DoS Hulk': 'DoS/DDoS',
        #         'DoS GoldenEye': 'DoS/DDoS',
        #         'DoS slowloris': 'DoS/DDoS',
        #         'DoS Slowhttptest': 'DoS/DDoS',
        #         'Heartbleed': 'DoS/DDoS',
        #         'FTP-Patator': 'Brute Force',
        #         'SSH-Patator': 'Brute Force',
        #         'Bot': 'Botnet ARES',
        #         'Web Attack =Brute Force': 'Web Attack',
        #         'Web Attack =Sql Injection': 'Web Attack',
        #         'Web Attack =XSS': 'Web Attack',
        #         'Infiltration': 'Infiltration'
        #     }

        # Create grouped label column
        if attack_group is not None:
            self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        else:
            self.data['label_category'] = self.data['label']
        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)

    def train_valid_test_split(self, seed=42):
        """"""

        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=seed,
            stratify=self.labels
        )
        if self.validation_size > 0:
            X_test, X_val, y_test, y_val = train_test_split(
                X_test,
                y_test,
                test_size=self.validation_size / (self.validation_size + self.testing_size),
                random_state=seed
            )
        else:
            X_val, y_val = None, None
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def scale(self, training_set, validation_set, testing_set):
        """"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set

        categorical_features = X_train.select_dtypes(exclude=["number"]).columns
        numeric_features = X_train.select_dtypes(exclude=[object]).columns

        preprocessor = ColumnTransformer(transformers=[
            # ('categoricals', OrdinalEncoder(), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
            # ('numericals', 'passthrough', numeric_features)
        ])
        preprocessor2 = ColumnTransformer(transformers=[

            # ('numericals', 'passthrough', numeric_features)
        ])

        # Preprocess the features
        columns_numeric = numeric_features.tolist()
        columns_categorical = categorical_features.tolist()

        print('数值列:', len(columns_numeric), columns_numeric)
        print('非数值列:', len(columns_categorical), columns_categorical)
        columns = columns_numeric # 全部的列

        print()

        # print(X_train.shape)
        # tmp = preprocessor.fit_transform(X_train)
        # print(tmp.shape)
        # exit(0)
        print(X_train.shape)
        print(X_test.shape)

        # 不能执行数值列的归一化

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=columns)

        # X_train = pd.DataFrame(preprocessor2.fit_transform(X_train), columns=columns)
        # X_test = pd.DataFrame(preprocessor2.fit_transform(X_test), columns=columns)

        # 获取生成的数值列名
        columns_numeric = numeric_features
        # 获取生成的所有列名
        columns = preprocessor.get_feature_names_out()

        # X_train = pd.DataFrame(preprocessor2.fit_transform(X_train), columns=columns_categorical)
        # X_test = pd.DataFrame(preprocessor2.fit_transform(X_test), columns=columns_categorical)

        # X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        # if X_val is not None:
        #     X_val = pd.DataFrame(preprocessor.fit_transform(X_val), columns=columns)
        # X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=columns)

        # 定义标签映射字典
        # 手动列出 'benign' 的12种大小写变体
        # 输入字符串
        import itertools

        input_str = "benign"

        # 获取所有可能的组合
        combinations = list(itertools.product(*[(char.lower(), char.upper()) for char in input_str]))

        # 将组合列表转换为字符串列表
        result = [''.join(combination) for combination in combinations]

        # 创建映射字典
        label_mapping = {combination: 0 for combination in result}

        # 输出结果
        for key in label_mapping:
            print(f"'{key}': {label_mapping[key]}")


        # 获取所有唯一的标签，并按字典序排序
        all_labels = sorted(set(y_train) | set(y_test))

        # 初始化计数器，跳过已经映射的 'benign' 和 'BENIGN'
        self.labels_encoded.append('benign')
        counter = 1
        for label in all_labels:
            if label not in label_mapping:
                label_mapping[label] = counter
                counter += 1
                self.labels_encoded.append(label)

        print(label_mapping)

        # 应用标签映射规则
        y_train = pd.DataFrame([label_mapping.get(label, 0) for label in y_train], columns=["label"])
        y_test = pd.DataFrame([label_mapping.get(label, 0) for label in y_test], columns=["label"])


        # 输出结果查看
        # print("y_train:")
        # print(y_train)
        # print("\ny_test:")
        # print(y_test)

        # 查看每个数字对应的原始label
        # self.labels_encoded.append('benign')
        # self.labels_encoded.append('malicious')


        # 获取label编码，len就是label数量
        # print(self.labels_encoded)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_data(dir_name, data_name, train_size, val_size, test_size, batch_size, seed, noise_type,
              corruption_ratio, sample_strategy,
              target_strategy):  # 路径名、数据集名称、训练集比例、验证集比例、测试集比例，0、1、2：欠采样、重采样、欠+重，0:中位数、1:平均数
    cicids2017 = CICIDS2017Preprocessor(
        data_name=data_name,
        data_path=dir_name,
        training_size=train_size,
        validation_size=val_size,
        testing_size=test_size
    )

    # Read datasets
    cicids2017.read_data()  # 读数据
    # cicids2017.print_data(1, 0)

    # Remove NaN, -Inf, +Inf, Duplicates
    cicids2017.remove_duplicate_values()  # 删除重复列
    cicids2017.remove_missing_values()  # 删除缺失值行
    cicids2017.remove_infinite_values()  # 删除无穷值
    # cicids2017.print_data(1)
    # Drop constant & correlated features
    cicids2017.remove_constant_features()  # 删除常量
    cicids2017.remove_correlated_features()  # 删除强相关变量
    # cicids2017.print_data(1)
    # Create new label category
    cicids2017.group_labels()  # 重定义label
    # cicids2017.print_data(5, 0)
    # cicids2017.find_non_numeric_columns()  # 去掉非数值列，不然后边的报错

    # Split & Normalise data sets
    training_set, validation_set, testing_set = cicids2017.train_valid_test_split(seed=seed)  # 训练集、测试集划分

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2017.scale(training_set, validation_set,
                                                                            testing_set)  # 归一化


    # print(type(X_train), type(y_train))

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # 将 y_train 转换为一维数组，并找出唯一值
    unique_labels, counts = np.unique(y_train, return_counts=True)

    # 输出唯一值和每个唯一值的出现次数
    print("Unique labels:", unique_labels)
    print("Counts:", counts)

    num_labels = len(unique_labels)
    train_set = np.concatenate((X_train, y_train), axis=1)
    test_set = np.concatenate((X_test, y_test), axis=1)

    from collections import Counter

    # 将 y_train 二维数组展平为一维数组
    y_train_flattened = y_train.reshape(-1)

    # 使用 Counter 统计每个类别的出现次数
    cnt_per_class_pre = Counter(y_train_flattened)

   

    # train_set = train_set[:, -33:]
    # test_set = test_set[:, -33:]

    # print(type(train_set), type(test_set))
    # print(train_set.dtype, test_set.dtype)
    # print(train_set.shape, test_set.shape)
    # print(train_set[:100])
    # print(test_set[:100])

    # 只用train去筛选
    combined_set = train_set

    np.random.seed(seed)
    np.random.shuffle(combined_set)

    # 分离标签为1和0的数据
    # if data_name == 'BoAu':
    #     benign_idx = 19
    # else:
    benign_idx = 0
    
    label_0_data = combined_set[combined_set[:, -1] == benign_idx] # ==0：良性
    label_1_data = combined_set[combined_set[:, -1] != benign_idx] # !=0：恶意

    print('良性:', label_0_data.shape)
    print('恶意:', label_1_data.shape)
    
    ratio = corruption_ratio

    if noise_type == 'asym':
        # 良性数据是不变的，恶意的翻转过来。
        # 0是良性，1是恶意
        selected_label_0_data = label_0_data
        
        sample_size = int(len(label_1_data) * ratio)
        # 选择相应数量的样本
        selected_label_1_data = label_1_data[:sample_size]

        # 将这些数据从原始数据集中移除，剩余数据是没有翻转的恶意数据
        remaining_ma_data = label_1_data[sample_size:]
        # 良性、翻转的恶意
        selected_label_0_data = np.concatenate((selected_label_0_data, selected_label_1_data), axis=0) # 这才是真正的be,带噪声的.

        # 合并两个数组
        be_ma_data = np.concatenate((selected_label_0_data, remaining_ma_data))

        print('拿到手的良性:', selected_label_0_data.shape)
        print('翻转的恶意:', selected_label_1_data.shape)
        print('剩余的恶意:', remaining_ma_data.shape) # 剩余的


        be_name = 'be_' + str(round(corruption_ratio, 1)) # 全部良性+翻转的恶意
        ma_name = 'ma_' + str(round(corruption_ratio, 1)) # 剩余的恶意
        be_ma_name = 'be_ma_' + str(round(corruption_ratio, 1)) # 二者加一起
        test_name = 'test_' + str(round(corruption_ratio, 1)) # 测试集


        # 构造完整的文件路径
        be_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{be_name}.npy'
        ma_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{ma_name}.npy'
        be_ma_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{be_ma_name}.npy' # 额外添加一个be_ma作为训练
        test_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{test_name}.npy'

        # 确保路径存在
        for file_path in [be_file_path, ma_file_path, be_ma_file_path, test_file_path]:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)

        # 保存数据到文件
        np.save(be_file_path, selected_label_0_data) # be是良性+恶意，其实
        np.save(ma_file_path, remaining_ma_data) # 剩余的这些恶意才是ma，不是翻转的，寄。虽然名字叫ma，其实是翻转之后剩下的ma，也就是remaining
        np.save(be_ma_file_path, be_ma_data)
        # np.save(remaining_file_path, remaining_data)
        np.save(test_file_path, test_set)
    else:
        
        # 复制原始数据集
        label_0_data_pre = label_0_data.copy()
        label_1_data_pre = label_1_data.copy()

        # 从label_0中一次性选取ratio比例的下标
        selected_indices_label_0 = np.random.choice(label_0_data_pre.shape[0], int(ratio * label_0_data_pre.shape[0]), replace=False)
        selected_data_label_0 = label_0_data_pre[selected_indices_label_0]

        # 将选中的数据均分成num_labels-1份，并修改标签
        split_data_label_0 = np.array_split(selected_data_label_0, num_labels - 1)

        # 创建存储label_1中改标签后的数据
        new_label_1_data = []

        for i in range(1, num_labels):
            split_data_label_0[i-1][:, -1] = i  # 修改标签为1到num_labels-1
            new_label_1_data.append(split_data_label_0[i-1])

        new_label_1_data = np.vstack(new_label_1_data)

        # 从label_1中选取相应比例的数据
        new_label_0_data = []

        for i in range(1, num_labels):
            current_class_data = label_1_data_pre[label_1_data_pre[:, -1] == i]
            selected_indices_label_1 = np.random.choice(current_class_data.shape[0], int(ratio * current_class_data.shape[0]), replace=False)
            selected_data_label_1 = current_class_data[selected_indices_label_1]
            selected_data_label_1[:, -1] = benign_idx  # 修改标签为0
            new_label_0_data.append(selected_data_label_1)

        new_label_0_data = np.vstack(new_label_0_data)

        # 更新label_0和label_1
        # label_0 = label_0_pre - selected_data_label_0 + new_label_0_data
        label_0_data = np.delete(label_0_data_pre, selected_indices_label_0, axis=0) # 删除翻转成恶意label的数据
        label_0_data = np.vstack((label_0_data, new_label_0_data)) # 添加翻转成良性label的数据

        # label_1 = label_1_pre - selected_data_label_1 + new_label_1_data
        for i in range(1, num_labels):
            current_class_data = label_1_data_pre[label_1_data_pre[:, -1] == i]
            current_class_indices = np.random.choice(current_class_data.shape[0], int(ratio * current_class_data.shape[0]), replace=False)
            label_1_data_pre = np.delete(label_1_data_pre, current_class_indices, axis=0) # 删除翻转成良性label的数据

        label_1_data = np.vstack((label_1_data_pre, new_label_1_data)) # 添加翻转成恶意label的数据

        # 结合处理后的label_0_data和label_1_data
        # final_combined_set = np.vstack((label_0_data, label_1_data))

        print('良性(带恶意噪声):', label_0_data.shape)
        print('恶意(带良性噪声):', label_1_data.shape)
        print('测试:', test_set.shape) # 剩余的


        be_name = 'be_' + str(round(corruption_ratio, 1)) # 全部良性+翻转的恶意
        ma_name = 'ma_' + str(round(corruption_ratio, 1)) # 剩余的恶意
        be_ma_name = 'be_ma_' + str(round(corruption_ratio, 1)) # 二者加一起
        test_name = 'test_' + str(round(corruption_ratio, 1)) # 测试集


        # 构造完整的文件路径
        be_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{be_name}.npy'
        ma_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{ma_name}.npy'
        test_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{test_name}.npy'
        be_ma_file_path = f'data/feat/{args.dataset_name}/{args.noise_type}/{be_ma_name}.npy' # 额外添加一个be_ma作为训练

        # 保存数据到文件
        np.save(be_file_path, label_0_data) # be是良性+恶意，其实
        np.save(ma_file_path, label_1_data) # 剩余的这些恶意才是ma，不是翻转的，寄。虽然名字叫ma，其实是翻转之后剩下的ma，也就是remaining
        # 合并两个数组
        be_ma_data = np.concatenate((label_0_data, label_1_data))
        np.save(be_ma_file_path, be_ma_data) #be_ma都存的
        np.save(test_file_path, test_set)

    # 将 cnt_per_class_pre 的结果写入到文本文件
    # with open('class_counts_pre.txt', 'w') as f:
    #     for class_value, count in cnt_per_class_pre.items():
    #         f.write(f"{count}\n")

    # # 将 cnt_per_class_fliped 的结果写入到另一个文本文件
    # with open('class_counts_fliped.txt', 'w') as f:
    #     for class_value, count in cnt_per_class_fliped.items():
    #         f.write(f"{count}\n")

    # Save the datasets to npy files
    

    # np.save('data/feat/{}.npy'.format(be_name), selected_label_0_data)
    # np.save('data/feat/{}.npy'.format(ma_name), selected_label_1_data)
    # np.save('data/feat/{}.npy'.format(test_name), remaining_data)

    # be_ma = np.concatenate((selected_label_1_data, selected_label_0_data), axis=0)
    # print(be_ma_file_path, ':', be_ma.shape)
    # Save the datasets to npy files
    # np.save(be_ma_file_path, be_ma)


    print("Datasets saved successfully.")




ratio_start = 0.0
ratio_distance = 0.2
ratio_end = 0.8
ratio_list = np.arange(ratio_start, ratio_end + ratio_distance, ratio_distance)
for ratio in ratio_list:
    load_data(f'data/dataset/{args.dataset_name}', args.dataset_name, 0.8, 0.0, 0.2, 128, 0, args.noise_type, ratio, 'none', 'none')
