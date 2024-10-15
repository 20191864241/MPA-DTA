import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
import re
import csv
import esm
import torch
from rdkit import Chem

# 生成蛋白质预训练表示:davis.npz,kiba.npz
def generate_protein_pretraining_representation(dataset, prots):
    data_dict = {}  # 数据字典
    prots_tuple = [(str(i), prots[i][:1022]) for i in range(len(prots))]  # 创建蛋白质元组
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()  # 加载transformer模型和字母表
    batch_converter = alphabet.get_batch_converter()  # 获取批处理转换器
    i = 0
    batch = 1

    while (batch*i) < len(prots):  # 循环处理蛋白质
        print('converting protein batch: '+ str(i))  # 打印转换蛋白质批次信息
        if (i + batch) < len(prots):  # 判断是否有下一个批次
            pt = prots_tuple[batch*i:batch*(i+1)]  # 获取当前批次的蛋白质元组
        else:
            pt = prots_tuple[batch*i:]  # 获取剩余的蛋白质元组

        batch_labels, batch_strs, batch_tokens = batch_converter(pt)  # 批量转换蛋白质
        #model = model.cuda()
        #batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)  # 获取结果
        token_representations = results["representations"][33].numpy()  # 提取表示
        data_dict[i] = token_representations  # 存储表示
        i += 1
    np.savez('data/proteins_' + dataset '.npz', dict=data_dict)  # 保存数据字典为npz文件

# 数据集处理
datasets = ['davis','kiba']
for dataset in datasets:
    proteins = json.load(open("proteins_"+dataset+".txt"), object_pairs_hook=OrderedDict)  # 加载蛋白质
    prots = []  # 蛋白质列表
    for t in proteins.keys():
        prots.append(proteins[t])

    # 生成蛋白质预训练表示
    generate_protein_pretraining_representation(dataset, prots)



