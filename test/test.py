import numpy as np

# 原始数据：形状 (5, 3, 4)
batch_size = 5
data = np.random.randn(batch_size, 3, 4)
print("原始数据形状:", data.shape)

# 创建更新掩码：哪些batch需要更新
update_mask = np.array([True, False, True, False, True])  # 第0,2,4个batch更新

# 新数据（只提供需要更新的batch）
new_data = np.random.randn(np.sum(update_mask), 3, 4)

# 部分更新
data[update_mask] = new_data