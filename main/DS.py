import os 
import sys

# import MADE.get_clean_epochs2 
sys.path.append('..')
import MADE
import Classifier
import AE

import argparse

# print(dir(MADE))



def generate(feat_dir, model_dir, made_dir, index, cuda):
    TRAIN_be = 'be_corrected'
    TRAIN_ma = 'ma_corrected'
    TRAIN = 'corrected'
    
    MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)

    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)

def generate_cpus(feat_dir, model_dir, made_dir, indices, cuda):
    for index in indices:
        generate(feat_dir, model_dir, made_dir, index, cuda)

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, corruption_ratio, epochs):
    
    # AE.train.main(data_dir, model_dir, cuda)
    # AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)
    # AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)
    # AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)
    TRAIN = 'be_ma_'
    TRAIN += str(round(corruption_ratio,1))
    print(TRAIN, '??')
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20', corruption_ratio, epochs, args.dataset_name, args.corruption_type)
    if args.corruption_type == 'asym':
        MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN, corruption_ratio, args.dataset_name, args.corruption_type)
    else:
        MADE.get_clean_epochs.main2(feat_dir, made_dir, '0.5', TRAIN, corruption_ratio, args.dataset_name, args.corruption_type)
    # MADE.get_clean_epochs2.main(feat_dir, made_dir, '0.5', TRAIN, corruption_ratio)
    # MADE.final_predict.main(feat_dir)
    
    # generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
    #
    # TRAIN = 'corrected'
    # Classifier.classify.main(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=5)
    
if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="根据命令行参数构造文件名。")

    # 添加 corruption_ratio 参数
    parser.add_argument('--dataset_name', type=str, default='CIC-IDS-2017')

    
    parser.add_argument('--corruption_type', type=str, default='asym')

    parser.add_argument('--corruption_ratio', type=float, default=0.7)


    # 添加 epochs 参数
    parser.add_argument('--epochs', type=int, default=100)

    # 解析命令行参数
    args = parser.parse_args()

    data_dir = '../data/data'
    feat_dir = '../data/feat/' + args.dataset_name + '/'  + args.corruption_type
    model_dir= '../data/model'
    made_dir = '../data/made'
    result_dir='../data/result'
    cuda = 0
    main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, args.corruption_ratio, args.epochs)

