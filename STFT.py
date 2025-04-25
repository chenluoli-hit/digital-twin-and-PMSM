import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
from scipy.signal.windows import dpss
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def load_and_process_data(file_path):
    """加载并预处理数据"""
    data = pd.read_csv(file_path)
    # 提取三相电流
    current_a = data['cDAQ1Mod2/ai0'].values
    current_b = data['cDAQ1Mod2/ai2'].values
    current_c = data['cDAQ1Mod2/ai3'].values
    return current_a, current_b, current_c


def apply_stft(signal, fs, window_type='hann', nperseg=512, noverlap=384, nfft=1024):
    """应用STFT并返回结果"""
    if window_type == 'dpss':
        # 创建DPSS窗(Slepian窗)
        # NW是时间带宽积参数，通常设置为2.5-4.0之间
        # Kmax是要计算的序列数量，通常取为NW*2-1的整数值
        NW = 3.5
        Kmax = 7
        window = dpss(nperseg, NW, Kmax)[0]  # 取第一个DPSS序列作为窗函数
    else:
        window = window_type

    f, t, Zxx = stft(signal, fs=fs,
                     window=window,
                     nperseg=nperseg,
                     noverlap=noverlap,
                     nfft=nfft,
                     boundary='zeros',
                     padded=True)

    # 计算幅值谱(dB)
    Sxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # 加上小值防止取对数时出现负无穷

    return f, t, Sxx_db


def plot_stft(f, t, Sxx_db, title, f_max=10000):
    """绘制STFT结果"""
    plt.figure(figsize=(10, 6))
    pcm = plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    plt.colorbar(pcm, label='amplitude (dB)')
    plt.ylabel('fs (Hz)')
    plt.xlabel('time (s)')
    plt.title(title)
    plt.ylim(0, f_max)
    plt.tight_layout()
    return plt


def compare_windows(signal, fs, nperseg=512, noverlap=384, nfft=1024):
    """比较不同窗函数的STFT结果"""
    window_types = ['hann', 'dpss']
    plt.figure(figsize=(15, 10))

    for i, window in enumerate(window_types):
        f, t, Sxx_db = apply_stft(signal, fs, window, nperseg, noverlap, nfft)

        plt.subplot(len(window_types), 1, i + 1)
        pcm = plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        plt.colorbar(pcm, label='amplitude (dB)')
        plt.ylabel('fs (Hz)')
        plt.xlabel('time (s)')
        plt.title(f'{window.upper()}  STFT')
        plt.ylim(0, 10000)

    plt.tight_layout()
    return plt


def extract_fault_features(Sxx_db, f, fundamental_freq=200):
    """提取故障特征

    根据电机匝间短路特性，主要关注基波频率的谐波和边带
    """
    # 找到基波频率对应的索引
    fundamental_idx = np.argmin(np.abs(f - fundamental_freq))

    # 提取基波分量的幅值
    fundamental_amp = np.mean(Sxx_db[fundamental_idx, :])

    # 提取2-5次谐波分量的幅值
    harmonic_amps = []
    for h in range(2, 6):
        h_idx = np.argmin(np.abs(f - fundamental_freq * h))
        harmonic_amps.append(np.mean(Sxx_db[h_idx, :]))

    # 计算边带频率(±20Hz和±40Hz)作为特征
    sideband_amps = []
    for offset in [20, 40]:
        upper_idx = np.argmin(np.abs(f - (fundamental_freq + offset)))
        lower_idx = np.argmin(np.abs(f - (fundamental_freq - offset)))

        upper_amp = np.mean(Sxx_db[upper_idx, :])
        lower_amp = np.mean(Sxx_db[lower_idx, :])

        sideband_amps.extend([upper_amp, lower_amp])

    # 提取频谱整体能量特征
    total_energy = np.sum(Sxx_db)
    low_freq_energy = np.sum(Sxx_db[f < 500, :])
    high_freq_energy = np.sum(Sxx_db[f >= 500, :])

    # 合并所有特征
    features = {
        'fundamental_amp': fundamental_amp,
        'harmonic_2nd': harmonic_amps[0],
        'harmonic_3rd': harmonic_amps[1],
        'harmonic_4th': harmonic_amps[2],
        'harmonic_5th': harmonic_amps[3],
        'upper_sideband_20Hz': sideband_amps[0],
        'lower_sideband_20Hz': sideband_amps[1],
        'upper_sideband_40Hz': sideband_amps[2],
        'lower_sideband_40Hz': sideband_amps[3],
        'total_energy': total_energy,
        'low_freq_energy': low_freq_energy,
        'high_freq_energy': high_freq_energy,
        'energy_ratio': high_freq_energy / (low_freq_energy + 1e-10)
    }

    return features


def process_dataset(root_folder='1.0kW/dataset', max_files=10):
    """处理整个数据集并提取特征"""
    # 故障等级文件夹: 0=健康, 1=轻微故障, 2=轻度故障, 3=中度故障, 4=重度故障
    folders = ['0', '1', '2', '3', '4']
    fault_labels = ['健康', '轻微故障', '轻度故障', '中度故障', '重度故障']

    all_features = []
    labels = []

    fs = 20000  # 采样率

    for folder_idx, folder in enumerate(folders):
        folder_path = os.path.join(root_folder, folder)
        files = os.listdir(folder_path)[:max_files]  # 限制文件数量加快处理

        print(f"处理 {fault_labels[folder_idx]} 数据...")

        for file_idx, file in enumerate(files):
            file_path = os.path.join(folder_path, file)

            # 加载和处理数据
            current_a, current_b, current_c = load_and_process_data(file_path)

            # 对三相电流分别应用STFT和特征提取
            for phase_idx, current in enumerate([current_a, current_b, current_c]):
                phase_name = ['A', 'B', 'C'][phase_idx]

                # 使用DPSS(Slepian)窗进行STFT
                f, t, Sxx_db = apply_stft(current, fs, window_type='dpss')

                # 提取故障特征
                features = extract_fault_features(Sxx_db, f)

                # 添加文件信息和相位信息
                features['file'] = file
                features['phase'] = phase_name

                # 保存特征和标签
                all_features.append(features)
                labels.append(folder_idx)

            if (file_idx + 1) % 5 == 0:
                print(f"  完成 {file_idx + 1}/{len(files)} 文件")

    # 转换为DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['label'] = labels
    features_df['label_name'] = [fault_labels[l] for l in labels]

    return features_df


def visualize_features(features_df):
    """可视化提取的特征"""
    # 选择数值型特征
    numeric_features = features_df.select_dtypes(include=[np.number])
    numeric_features = numeric_features.drop(['label'], axis=1)

    # 特征标准化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_features.columns)

    # 添加标签
    scaled_df['label_name'] = features_df['label_name']

    # 绘制箱线图
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features.columns[:6]):  # 显示前6个特征
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='label_name', y=feature, data=scaled_df)
        plt.xticks(rotation=45)
        plt.title(f'特征: {feature}')

    plt.tight_layout()
    plt.show()

    # 绘制热图展示特征间相关性
    plt.figure(figsize=(12, 10))
    corr = numeric_features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
                fmt='.2f', square=True, linewidths=.5)
    plt.title('特征相关性热图')
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 设置样本参数
    fs = 20000  # 采样率Hz

    # 1. 单文件分析示例
    print("执行单文件STFT分析...")
    file_path = '1.0kW/dataset/2/1000W_fault_2_segment_0.csv'  # 轻微故障示例
    current_a, current_b, current_c = load_and_process_data(file_path)

    # 使用Hann窗和DPSS(Slepian)窗比较分析
    plt_compare = compare_windows(current_a, fs)
    plt_compare.suptitle('current - Hann compare DPSS(Slepian)')
    plt_compare.show()

    # 对三相电流分别进行STFT分析
    for i, (current, phase) in enumerate(zip([current_a, current_b, current_c], ['A', 'B', 'C'])):
        f, t, Sxx_db = apply_stft(current, fs, window_type='dpss')
        plt = plot_stft(f, t, Sxx_db, f'{phase}current STFT (DPSS)')
        plt.show()

        # 提取并打印该相位的故障特征
        features = extract_fault_features(Sxx_db, f)
        print(f"\n{phase}相电流故障特征:")
        for key, value in features.items():
            print(f"{key}: {value:.4f}")

    # 2. 多文件特征提取与分析(可选，处理时间可能较长)
    process_full_dataset = input("\n是否处理整个数据集进行特征提取? (y/n): ")
    if process_full_dataset.lower() == 'y':
        print("\n开始处理数据集...")
        # 为了加快示例运行，每个类别只处理5个文件
        features_df = process_dataset(max_files=5)

        # 保存提取的特征
        features_df.to_csv('pmsm_fault_features.csv', index=False)
        print(f"特征已保存到 'pmsm_fault_features.csv'")

        # 可视化特征
        visualize_features(features_df)