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

    
if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="根据命令行参数构造文件名。")

    # 添加 noise_ratio 参数
    parser.add_argument('--dataset_name', type=str, default='CIC-IDS-2017')

    
    parser.add_argument('--noise_type', type=str, default='asym')

    parser.add_argument('--noise_ratio', type=float, default=0.7)


    # 添加 epochs 参数
    parser.add_argument('--epochs', type=int, default=100)

    # 解析命令行参数
    args = parser.parse_args()


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
    main(model_dir, feat_dir, made_dir, result_dir, cuda, args.noise_ratio, args.epochs)


# print(dir(MADE))

# 创建目录的函数
def ensure_dir(directory):
    print('ok')
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def main(model_dir, feat_dir, made_dir, result_dir, cuda, noise_ratio, epochs):
    
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

