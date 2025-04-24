import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import PchipInterpolator, CubicSpline
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 参数设置
input_file = '1.0kW/testdata_csv/1000W_fault_1.csv'
output_file = '1.0kW/testdata_csv/1000W_fault_1_savgol.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
chunksize = 500000
fs = 100000  # 原始采样频率，Hz
cutoff = 10000  # 截止频率，Hz
target_fs = 20000  # 目标采样频率，Hz
decimation_factor = int(fs / target_fs)  # 降采样因子
current_columns = ['cDAQ1Mod2/ai0', 'cDAQ1Mod2/ai2', 'cDAQ1Mod2/ai3']

# 插值方法选择: 'pchip', 'cubic', 'akima', 'savgol'
interpolation_method = 'akima'


# 低通滤波器函数
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# 高级插值函数
def advanced_interpolate(data, method='akima'):
    """
    使用高级插值方法处理数据中的缺失值
    """
    # 创建一份数据副本
    interpolated_data = data.copy()

    # 获取非缺失值的索引和值
    valid_indices = np.where(~pd.isna(data))[0]
    valid_values = data.iloc[valid_indices]

    if len(valid_indices) < 4:  # 如果有效点太少，回退到线性插值
        return data.interpolate(method='linear')

    # 处理所有缺失值
    missing_indices = np.where(pd.isna(data))[0]

    if len(missing_indices) == 0:
        return data  # 没有缺失值，直接返回

    if method == 'pchip':
        # PCHIP插值
        pchip = PchipInterpolator(valid_indices, valid_values)
        interpolated_data.iloc[missing_indices] = pchip(missing_indices)

    elif method == 'cubic':
        # 三次样条插值
        cs = CubicSpline(valid_indices, valid_values)
        interpolated_data.iloc[missing_indices] = cs(missing_indices)

    elif method == 'akima':
        # 尝试导入Akima插值
        try:
            from scipy.interpolate import Akima1DInterpolator
            akima = Akima1DInterpolator(valid_indices, valid_values)
            interpolated_data.iloc[missing_indices] = akima(missing_indices)
        except (ImportError, ValueError):
            print("Akima插值不可用，回退到PCHIP")
            pchip = PchipInterpolator(valid_indices, valid_values)
            interpolated_data.iloc[missing_indices] = pchip(missing_indices)

    elif method == 'savgol':
        # 先用线性插值填充缺失值
        temp_data = data.interpolate(method='linear')
        # 然后应用Savitzky-Golay滤波进行平滑
        window_length = min(51, len(data) // 4 * 2 + 1)  # 确保窗口长度为奇数且不超过数据长度的四分之一
        window_length = max(window_length, 5)  # 至少为5
        poly_order = min(3, window_length - 1)  # 多项式阶数
        smoothed = savgol_filter(temp_data, window_length, poly_order)
        return pd.Series(smoothed, index=data.index)

    else:
        # 默认回退到线性插值
        return data.interpolate(method='linear')

    return interpolated_data


# 数据处理函数
def process_chunk(chunk):
    # 保存原始时间格式
    original_time_format = chunk['Time'].dtype

    # 如果需要处理时间列，获取第一个时间值和时间间隔
    first_time = chunk['Time'].iloc[0]

    original_length = len(chunk)
    result_chunk = chunk.copy()

    # 对异常值进行处理 - 不再删除行，而是替换为NaN后插值
    for col in current_columns:
        z_scores = stats.zscore(result_chunk[col])
        # 将异常值标记为NaN而不是删除行
        result_chunk.loc[abs(z_scores) > 3, col] = np.nan

    # 使用插值填充NaN值
    for col in current_columns:
        result_chunk[col] = advanced_interpolate(result_chunk[col], method=interpolation_method)

    # 低通滤波
    for col in current_columns:
        result_chunk[col] = butter_lowpass_filter(result_chunk[col], cutoff, fs)

    # 降采样 - 保持预期的行数
    result_chunk = result_chunk.iloc[::decimation_factor, :].reset_index(drop=True)

    # 正确处理时间戳
    if pd.api.types.is_numeric_dtype(original_time_format):
        # 如果是数值型时间，按照降采样后的间隔重新生成
        time_interval = decimation_factor / fs  # 降采样后的时间间隔
        chunk['Time'] = first_time + np.arange(len(chunk)) * time_interval
    else:
        # 尝试按照日期时间格式处理
        try:
            chunk['Time'] = pd.date_range(
                start=first_time,
                periods=len(chunk),
                freq=f'{1 / target_fs * 1e6}us'
            )
        except:
            # 如果失败，回退到数值方法
            time_interval = decimation_factor / fs
            chunk['Time'] = first_time + np.arange(len(chunk)) * time_interval

    return result_chunk


# 清除已有的输出文件
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"已删除现有的输出文件: {output_file}")

# 获取原始文件的总行数
try:
    original_total_rows = sum(len(chunk) for chunk in pd.read_csv(input_file, chunksize=chunksize))
    print(f"原始文件总行数: {original_total_rows}")
    expected_rows_after_decimation = original_total_rows // decimation_factor
    print(f"预期降采样后行数: {expected_rows_after_decimation}")
except Exception as e:
    print(f"无法读取原始文件总行数: {e}")
    original_total_rows = None

# 分块处理并保存
processed_total_rows = 0
for chunk in pd.read_csv(input_file, chunksize=chunksize):
    processed_chunk = process_chunk(chunk)
    processed_total_rows += len(processed_chunk)
    processed_chunk.to_csv(output_file, mode='a',
                           header=not os.path.exists(output_file),
                           index=False)
    print(f"已处理 {len(processed_chunk)} 行数据，当前总行数: {processed_total_rows}")

# 可视化
print("\n生成数据清洗前后的比较图...")
original_data = pd.read_csv(input_file, nrows=5000)
cleaned_data = pd.read_csv(output_file, nrows=1000)

# 打印时间戳样本以检查
print("原始数据时间戳示例:")
print(original_data['Time'].iloc[:5])
print("\n处理后数据时间戳示例:")
print(cleaned_data['Time'].iloc[:5])

fig, axes = plt.subplots(len(current_columns), 2, figsize=(15, 10))
fig.suptitle(f'compare (method: {interpolation_method})', fontsize=16)

for i, col in enumerate(current_columns):
    axes[i, 0].plot(original_data[col].iloc[:1000], color='blue')
    axes[i, 0].set_title(f'{col} - before-clean')
    axes[i, 0].set_ylabel('current')
    #axes[i, 0].grid(True, linestyle='--', alpha=0.7)

    cleaned_sample = cleaned_data.iloc[:min(200, len(cleaned_data))]
    axes[i, 1].plot(cleaned_sample[col], color='green')
    axes[i, 1].set_title(f'{col} - after-clean')
    #axes[i, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('1.0kW/testdata_csv/akima.png')
#plt.savefig('1.0kW/testdata_csv/pchip.png', dpi=300)
print("已保存可视化结果到 akima.png")

# 打印摘要
print("\n数据处理结果摘要:")
if original_total_rows:
    print(f"原始数据总行数: {original_total_rows}")
    print(f"预期降采样后行数: {expected_rows_after_decimation}")
print(f"实际处理后数据总行数: {processed_total_rows}")
print(f"插值方法: {interpolation_method}")
print("数据处理完成!")