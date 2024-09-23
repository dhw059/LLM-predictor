import os
import sys

# 设置文件路径
train_path = "data/samples/textedge_prop_mp22_train.csv"
valid_path = "data/samples/textedge_prop_mp22_valid.csv"
test_path = "data/samples/textedge_prop_mp22_test.csv"
epochs = 2  # 默认的 epoch 数是 200 以获得最佳性能
property_name = "band_gap"  # 指定属性名称

# 构建 Python 命令行参数
command = [
    'D:\ProgramData\envs\LLM\python.exe',  # 当前 Python 解释器的路径
    'python', 'llmprop_train.py',  # 要运行的 Python 脚本
    '--train_data_path', train_path,
    '--valid_data_path', valid_path,
    '--test_data_path', test_path,
    '--epochs', str(epochs),
    '--property', property_name
]

# 运行命令
os.system(' '.join(command))