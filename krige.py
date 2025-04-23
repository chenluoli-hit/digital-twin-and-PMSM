import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# 定义异常值移除函数（使用 Z-score 方法）
def remove_outliers(chunk, columns, threshold=100):
    chunk_copy = chunk.copy()
    for col in columns:
        # 提取列数据并丢弃 NaN
        col_data = chunk_copy[col].dropna()
        if len(col_data) > 0:  # 确保数据不为空
            # 计算 Z-score
            z_scores = np.abs(stats.zscore(col_data))
            # 获取异常值的条件
            outlier_condition = z_scores > threshold
            # 将条件映射回原始数据的索引
            outlier_indices = col_data.index[outlier_condition]
            # 将异常值替换为 NaN
            chunk_copy.loc[outlier_indices, col] = np.nan
    return chunk_copy


# 文件路径
input_file = '1.0kW/testdata_csv/1000W_fault_1.csv'
output_file = '1.0kW/testdata_csv/1000W_fault_1_cleaned.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 定义要处理的列
current_columns = ['cDAQ1Mod2/ai0', 'cDAQ1Mod2/ai2', 'cDAQ1Mod2/ai3']  # 确保这些列名与文件中一致

# 分块大小
chunksize = 10000

# 存储处理后的数据块
cleaned_chunks = []

# 分块处理
for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
    print(f"处理数据块 {i + 1}...")

    # 移除异常值（使用 Z-score 方法）
    chunk_cleaned = remove_outliers(chunk, current_columns, threshold=100)

    # 确保数值类型
    for col in current_columns:
        if col in chunk.columns:  # 检查列是否存在
            if chunk_cleaned[col].dtype == 'object':
                chunk_cleaned[col] = pd.to_numeric(chunk_cleaned[col], errors='coerce')
        else:
            print(f"警告：列 {col} 在当前数据块中缺失")
            continue

    # 插值和平滑
    for col in current_columns:
        if col in chunk_cleaned.columns:
            # 提取有效数据点
            valid_idx = chunk_cleaned[col].dropna().index
            x = valid_idx.values.astype(float)  # 时间索引作为空间坐标[5](@ref)
            z = chunk_cleaned.loc[valid_idx, col].values

            if len(z) > 10:  # 确保足够数据点建立变差函数
                # 构建克里金模型（一维时间序列）
                OK = OrdinaryKriging(
                    x,
                    np.zeros_like(x),  # 伪二维坐标（时间维度）
                    z,
                    variogram_model='exponential',  # 推荐指数模型[1](@ref)
                    coordinates_type='euclidean',
                    verbose=False
                )

                # 生成预测网格（完整时间序列）
                grid_x = chunk_cleaned.index.values.astype(float)
                grid_y = np.zeros_like(grid_x)

                # 执行插值
                z_pred, _ = OK.execute('points', grid_x, grid_y)
                chunk_cleaned[col] = z_pred
            else:
                # 数据不足时退用线性插值
                chunk_cleaned[col] = chunk_cleaned[col].interpolate(method='linear')

            # 保留平滑处理
            chunk_cleaned[col] = chunk_cleaned[col].rolling(window=5, center=True).mean()
            chunk_cleaned[col] = chunk_cleaned[col].ffill().bfill()

    cleaned_chunks.append(chunk_cleaned)

# 合并数据
print("合并所有处理后的数据...")
cleaned_data = pd.concat(cleaned_chunks, ignore_index=True)

# 保存结果
print(f"保存清洗后的数据到 {output_file}")
cleaned_data.to_csv(output_file, index=False)

# 可视化
print("生成数据清洗前后的比较图...")
original_data = pd.read_csv(input_file, nrows=1000)
fig, axes = plt.subplots(len(current_columns), 2, figsize=(15, 10))
fig.suptitle('compare', fontsize=16)

for i, col in enumerate(current_columns):
    axes[i, 0].plot(original_data[col], color='blue')
    axes[i, 0].set_title(f'{col} - before-clean')
    axes[i, 0].set_ylabel('current')

    cleaned_sample = cleaned_data.iloc[:min(1000, len(cleaned_data))]
    axes[i, 1].plot(cleaned_sample[col], color='green')
    axes[i, 1].set_title(f'{col} - after-clean')

    axes[i, 1].text(0.05, 0.95,
                f'Kriging Model: {OK.variogram_model}',
                transform=axes[i, 1].transAxes,
                fontsize=9, verticalalignment='top')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('1.0kW/testdata_csv/cleaning_comparison.png')
print("已保存可视化结果到 cleaning_comparison.png")

# 打印摘要
print("\n数据清洗结果摘要:")
print(f"原始数据总行数: {sum(1 for _ in pd.read_csv(input_file, chunksize=1000))}")
print(f"清洗后数据总行数: {len(cleaned_data)}")
print("数据清洗完成!")