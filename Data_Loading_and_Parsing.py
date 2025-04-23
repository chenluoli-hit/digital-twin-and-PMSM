# write by chen ll
# 2025.4.23
# welcome to email 1292593513@qq.com
from nptdms import TdmsFile
import numpy as np
import os
import glob
import pandas as pd
# 加载 TDMS 文件
tdms_file = TdmsFile.read("1.0kW/current/1000W_health_0.tdms")

# 列出所有组
groups = tdms_file.groups()
print("Groups:", [group.name for group in groups])

# 数据在第二个组中，列出该组的所有通道
group = groups[1]
channels = group.channels()
print("Channels:", [channel.name for channel in channels])

# 获取通道属性（以第一个通道为例）
channel = group.channels()[0]
channel_properties = channel.properties
print("Channel Properties:", channel_properties)

# 从属性里取出时间增量（单位：秒）
wf_inc = channel_properties.get('wf_increment', None)
if wf_inc is None:
    print("属性里没有 'wf_increment'，请确认通道是否正确")
else:
    # 采样率 = 1 / 时间增量
    sampling_rate = 1.0 / wf_inc
    print(f"Sampling Rate: {sampling_rate:.0f} Hz")

# 打上时间戳
channel = group.channels()[0]
props = channel.properties

dt = props['wf_increment']     # 时间步长，单位：秒
n_samples = props['wf_samples']  # 总采样点数

# 构造时间戳数组，从 0 开始，每隔 dt 秒，共 n_samples 个
time_stamps = np.arange(n_samples) * dt

print("前 5 个时间戳：", time_stamps[:5])

# 获取第一个通道的数据长度
data_length = len(channel)
print("Data Length:", data_length)

# 检查所有通道的长度是否一致
for channel in group.channels():
    print(f"Channel: {channel.name}, Length: {len(channel)}")

# 指定你的数据目录
data_dir = r"D:\pycharm\myfinish\1.0kW\current"

# 找到所有 .tdms 文件
tdms_files = glob.glob(os.path.join(data_dir, "*.tdms"))

file_labels = {}
for fp in tdms_files:
    base_name = os.path.basename(fp)                   # e.g. "1000W_health_0.tdms"
    name_no_ext, _ = os.path.splitext(base_name)       # e.g. "1000W_health_0"
    label_str = name_no_ext.split('_')[-1]             # 取最后一段 "0"
    label = int(label_str)                             # 转成整数
    file_labels[fp] = label
    print(f"{base_name} → label = {label}")

# 假设数据在第一个组中
group = tdms_file.groups()[1]

# 读取三相电流数据（假设通道名为 'Ia', 'Ib', 'Ic'）
currents = ['cDAQ1Mod2/ai0', 'cDAQ1Mod2/ai2', 'cDAQ1Mod2/ai3']

for fp in tdms_files:
    # 1) 读取 TDMS 文件
    tdms_file = TdmsFile.read(fp)
    group = tdms_file.groups()[1]     # 第二个组

    # 2) 取出三相电流数据
    data = {ch: group[ch][:] for ch in currents}
    df = pd.DataFrame(data)

    # 3) 打标签
    label = file_labels[fp]           # 0–4
    df['Label'] = label

    # （可选）加上文件名，方便区分
    df['Filename'] = os.path.basename(fp)

    # 4) 打印前 5 行
    print(f"===== {os.path.basename(fp)} (Label={label}) =====")
    print(df.head(), "\n")
