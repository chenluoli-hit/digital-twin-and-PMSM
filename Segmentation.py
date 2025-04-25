import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
from tqdm import tqdm  # 用于显示进度条

# 全局参数
WINDOW_SIZE = 300  # 每个窗口包含的数据点数
STEP_SIZE = 150  # 窗口之间的步长
SOURCE_FOLDER = '1.0kW/clean_data_csv'  # 源数据文件夹
OUTPUT_FOLDER = '1.0kW/clean_data_csv'  # 输出数据文件夹
PHASE_COLUMNS = ['cDAQ1Mod2/ai0', 'cDAQ1Mod2/ai2', 'cDAQ1Mod2/ai3']  # 三相电流列名

# 确保输出文件夹存在
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_data(file_path):
    """从CSV文件加载数据，返回电流数据和标签"""
    try:
        data = pd.read_csv(file_path)
        # 检查必要的列是否存在
        required_columns = PHASE_COLUMNS + ['Label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"警告: 文件 {file_path} 缺少以下列: {missing_columns}")
            return None, None

        currents = data[PHASE_COLUMNS].values
        labels = data['Label'].values
        return currents, labels
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None, None


def segment_data(currents, labels, window_size, step_size):
    """将连续的电流数据分割成固定长度的窗口"""
    num_samples = (currents.shape[0] - window_size) // step_size + 1
    segments = np.zeros((num_samples, window_size, currents.shape[1]))
    segment_labels = np.zeros(num_samples)

    for i in range(num_samples):
        start = i * step_size
        end = start + window_size
        segments[i] = currents[start:end, :]
        # 使用窗口中最后一个点的标签作为整个窗口的标签
        segment_labels[i] = labels[end - 1]

    return segments, segment_labels


def standardize_segments(segments):
    """对每个窗口进行Z-Score标准化"""
    # 预分配内存，提高效率
    standardized_segments = np.zeros_like(segments)

    for i, segment in enumerate(segments):
        mean = np.mean(segment, axis=0)
        std = np.std(segment, axis=0)
        # 避免除以零
        std = np.where(std == 0, 1e-10, std)
        standardized_segments[i] = (segment - mean) / std

    return standardized_segments


def min_max_normalize_segments(segments, feature_range=(0, 1)):
    """对每个窗口进行Min-Max归一化"""
    min_val, max_val = feature_range
    normalized_segments = np.zeros_like(segments)

    for i, segment in enumerate(segments):
        seg_min = np.min(segment, axis=0)
        seg_max = np.max(segment, axis=0)
        # 避免除以零
        denominator = seg_max - seg_min
        denominator = np.where(denominator == 0, 1e-10, denominator)

        normalized_segments[i] = min_val + ((segment - seg_min) / denominator) * (max_val - min_val)

    return normalized_segments


def preprocess_file(file_path, window_size, step_size, normalization='z_score'):
    """预处理单个文件"""
    currents, labels = load_data(file_path)
    if currents is None:
        return None, None

    segments, segment_labels = segment_data(currents, labels, window_size, step_size)

    if normalization == 'z_score':
        normalized_segments = standardize_segments(segments)
    elif normalization == 'min_max':
        normalized_segments = min_max_normalize_segments(segments)
    else:
        raise ValueError(f"不支持的归一化方法: {normalization}")

    return normalized_segments, segment_labels


def plot_segment_sample(segments, labels, index=0, save_path=None):
    """绘制并保存示例片段的三相电流波形"""
    plt.figure(figsize=(12, 6))
    for i, phase in enumerate(['Phase A', 'Phase B', 'Phase C']):
        plt.plot(segments[index, :, i], label=phase)
    plt.title(f'Segment {index} (Label: {labels[index]})')
    plt.xlabel('Time Point')
    plt.ylabel('Normalized Current')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def process_all_files(source_folder, output_folder, window_size, step_size, normalization='z_score'):
    """处理文件夹中的所有CSV文件"""
    # 获取所有CSV文件
    csv_files = list(Path(source_folder).glob('*.csv'))

    if not csv_files:
        print(f"在 {source_folder} 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，开始处理...")

    # 统计信息
    total_segments = 0
    start_time = time.time()

    # 使用tqdm显示进度
    for file_path in tqdm(csv_files, desc="处理文件"):
        try:
            # 从文件名提取基本名称
            base_filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
            output_filename = f"{filename_without_ext}_pre"

            # 预处理数据
            segments, labels = preprocess_file(file_path, window_size, step_size, normalization)

            if segments is None:
                continue

            # 保存处理后的数据
            np.savez(
                os.path.join(output_folder, f"{output_filename}.npz"),
                segments=segments,
                labels=labels
            )

            # 保存第一个片段的可视化图
            plot_segment_sample(
                segments,
                labels,
                index=0,
                save_path=os.path.join(output_folder, f"{output_filename}_sample.png")
            )

            # 创建CSV文件保存处理信息
            info_df = pd.DataFrame({
                'original_file': [base_filename],
                'segments_count': [segments.shape[0]],
                'window_size': [window_size],
                'step_size': [step_size],
                'normalization': [normalization],
                'unique_labels': [np.unique(labels).tolist()]
            })
            info_df.to_csv(os.path.join(output_folder, f"{output_filename}_info.csv"), index=False)

            total_segments += segments.shape[0]

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"处理完成! 总共处理了 {len(csv_files)} 个文件，生成了 {total_segments} 个片段")
    print(f"总处理时间: {processing_time:.2f} 秒")


def main():
    """主函数"""
    # 可以在这里修改参数
    normalization_method = 'z_score'  # 'z_score' 或 'min_max'

    process_all_files(
        source_folder=SOURCE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        normalization=normalization_method
    )

    print("所有数据预处理完成!")


if __name__ == "__main__":
    main()