import numpy as np

# node_proteins = np.load('data/edge/davis_edge.npz', allow_pickle=True)
# arrays_in_file = node_proteins.files
# for array_name in arrays_in_file:
#     print(f"Array name: {array_name}, shape: {node_proteins[array_name].shape}")
#
proteins = np.load('davis.npz', allow_pickle=True)
proteins = proteins['dict'][()]
print(proteins.keys())
# for key in proteins.keys():
#     print(f"Key: {key}, Shape: {proteins[key].shape}")

# import numpy as np
#
# def pad_tensor(tensor, target_shape):
#     """
#     将张量从 (1, N, 1280) 填充为 (1, 1024, 1280)
#     """
#     padding_size = target_shape[1] - tensor.shape[1]
#     padded_tensor = np.pad(tensor, ((0, 0), (0, padding_size), (0, 0)), mode='constant')
#     return padded_tensor
#
# # 加载数据
# proteins = np.load('davis.npz', allow_pickle=True)
# proteins = proteins['dict'][()]
#
# # 目标形状
# target_shape = (1, 1024, 1280)
#
# # 创建新的字典以存储填充后的张量
# padded_proteins = {}
#
# # 遍历并填充每个张量
# for key in proteins.keys():
#     padded_proteins[key] = pad_tensor(proteins[key], target_shape)
#     print(padded_proteins[key].shape)
#     # 保存到新的 .npz 文件
#     np.savez('davis_new.npz', dict=padded_proteins)

