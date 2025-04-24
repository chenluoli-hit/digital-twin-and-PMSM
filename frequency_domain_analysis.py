import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import savgol_filter
import argparse


def calculate_thd(spectrum, fundamental_idx):
    """
    计算总谐波失真(THD)

    参数:
    - spectrum: FFT计算出的幅值谱
    - fundamental_idx: 基波的索引位置

    返回:
    - THD值 (百分比)
    """
    fundamental_amplitude = spectrum[fundamental_idx]
    if fundamental_amplitude == 0:
        return float('inf')  # 避免除以零

    # 计算所有谐波的平方和
    harmonic_sum_squared = 0
    for i in range(2, len(spectrum) // 2):  # 从2次谐波开始
        if i * fundamental_idx < len(spectrum):
            harmonic_amplitude = spectrum[i * fundamental_idx]
            harmonic_sum_squared += harmonic_amplitude ** 2

    thd = np.sqrt(harmonic_sum_squared) / fundamental_amplitude * 100
    return thd


def calculate_band_energy(spectrum, freq_values, start_freq, end_freq):
    """
    计算指定频带内的能量

    参数:
    - spectrum: FFT计算出的幅值谱
    - freq_values: 对应的频率值
    - start_freq: 频带起始频率
    - end_freq: 频带结束频率

    返回:
    - 该频带内的能量
    """
    mask = (freq_values >= start_freq) & (freq_values <= end_freq)
    band_energy = np.sum(np.square(spectrum[mask]))
    return band_energy


def analyze_frequency_domain(data_file, sampling_rate, current_columns, plot=True, output_dir='results'):
    """
    对电流数据进行频域特征分析

    参数:
    - data_file: 电流数据文件路径
    - sampling_rate: 采样率 (Hz)
    - current_columns: 电流数据的列名列表
    - plot: 是否生成并保存频谱图
    - output_dir: 结果输出目录

    返回:
    - 频域特征字典
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在分析文件: {data_file}")

    # 读取数据
    df = pd.read_csv(data_file)

    # 确保存在时间列
    if 'Time' not in df.columns:
        time_column = None
    else:
        time_column = 'Time'

    # 分析每个电流通道
    results = {}

    for column in current_columns:
        if column not in df.columns:
            print(f"警告: 列 '{column}' 在数据文件中不存在，将跳过")
            continue

        print(f"\n分析电流通道: {column}")
        current_data = df[column].values

        # 移除直流分量 (均值)
        current_data = current_data - np.mean(current_data)

        # 应用窗函数减少频谱泄露
        window = np.hanning(len(current_data))
        windowed_data = current_data * window

        # 执行FFT
        n = len(windowed_data)
        yf = fft(windowed_data)
        xf = fftfreq(n, 1 / sampling_rate)

        # 只取正频率部分
        xf = xf[:n // 2]
        yf_abs = 2.0 / n * np.abs(yf[:n // 2])

        # 寻找基波频率 (最大幅值处，不考虑DC和非常低频部分)
        # 假设基波频率高于5Hz
        min_freq_idx = np.where(xf >= 5)[0][0]
        fundamental_idx = min_freq_idx + np.argmax(yf_abs[min_freq_idx:])
        fundamental_freq = xf[fundamental_idx]
        fundamental_amp = yf_abs[fundamental_idx]

        print(f"基波频率: {fundamental_freq:.2f} Hz")
        print(f"基波幅值: {fundamental_amp:.4f}")

        # 获取指定谐波幅值
        harmonics = [3, 5, 7, 9, 11]
        harmonic_amps = {}

        for h in harmonics:
            # 寻找接近整数倍谐波的频率
            h_freq = h * fundamental_freq
            h_idx = np.argmin(np.abs(xf - h_freq))
            h_amp = yf_abs[h_idx]
            harmonic_amps[h] = h_amp
            print(f"{h}次谐波 ({xf[h_idx]:.2f} Hz) 幅值: {h_amp:.4f}")

        # 计算总谐波失真
        thd = calculate_thd(yf_abs, fundamental_idx)
        print(f"总谐波失真 (THD): {thd:.2f}%")

        # 计算不同频带能量
        band_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
        band_energies = {}

        for start, end in band_ranges:
            energy = calculate_band_energy(yf_abs, xf, start, end)
            band_energies[(start, end)] = energy
            print(f"频带 {start}-{end} Hz 能量: {energy:.4f}")

        # 计算负序分量（需要三相电流才能计算）
        # 此处略过，如果有三相电流数据可以添加

        # 存储结果
        results[column] = {
            'fundamental_freq': fundamental_freq,
            'fundamental_amp': fundamental_amp,
            'harmonics': harmonic_amps,
            'thd': thd,
            'band_energies': band_energies,
            'spectrum': {
                'freq': xf,
                'amp': yf_abs
            }
        }

        # 绘制频谱图
        if plot:
            plt.figure(figsize=(14, 8))

            # 线性尺度
            plt.subplot(211)
            plt.plot(xf, yf_abs)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('频率 (Hz)')
            plt.ylabel('幅值')
            plt.title(f'{column} - 线性频谱')

            # 标记基波和谐波
            plt.axvline(x=fundamental_freq, color='r', linestyle='-', alpha=0.5,
                        label=f'基波 ({fundamental_freq:.1f} Hz)')
            for h in harmonics:
                h_freq = h * fundamental_freq
                h_idx = np.argmin(np.abs(xf - h_freq))
                plt.axvline(x=xf[h_idx], color='g', linestyle='--', alpha=0.5)

            # 对数尺度 (更好地显示谐波)
            plt.subplot(212)
            plt.semilogy(xf, yf_abs)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('频率 (Hz)')
            plt.ylabel('幅值 (对数)')
            plt.title(f'{column} - 对数频谱')

            # 标记基波和谐波
            plt.axvline(x=fundamental_freq, color='r', linestyle='-', alpha=0.5,
                        label=f'基波 ({fundamental_freq:.1f} Hz)')
            for h in harmonics:
                h_freq = h * fundamental_freq
                h_idx = np.argmin(np.abs(xf - h_freq))
                plt.axvline(x=xf[h_idx], color='g', linestyle='--', alpha=0.5,
                            label=f'{h}次谐波' if h == harmonics[0] else "")

            plt.legend()
            plt.tight_layout()

            # 保存图像
            filename = os.path.join(output_dir,
                                    f'spectrum_{os.path.basename(data_file).split(".")[0]}_{column.replace("/", "_")}.png')
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"频谱图已保存到: {filename}")

    return results


def compare_interpolation_methods(original_file, interpolated_files, method_names, sampling_rate, current_columns):
    """
    比较不同插值方法的频域特征

    参数:
    - original_file: 原始数据文件
    - interpolated_files: 使用不同插值方法处理后的文件列表
    - method_names: 对应的插值方法名称列表
    - sampling_rate: 采样率
    - current_columns: 电流列名列表
    """
    # 创建结果目录
    output_dir = 'frequency_analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分析原始数据
    print("分析原始数据...")
    original_results = analyze_frequency_domain(original_file, sampling_rate, current_columns, plot=True,
                                                output_dir=output_dir)

    # 分析每种插值方法处理后的数据
    all_results = [original_results]
    for file, method in zip(interpolated_files, method_names):
        print(f"\n分析 {method} 插值方法处理后的数据...")
        results = analyze_frequency_domain(file, sampling_rate, current_columns, plot=True, output_dir=output_dir)
        all_results.append(results)

    # 比较结果并生成汇总图表
    compare_and_visualize_results(all_results, ['原始数据'] + method_names, current_columns, output_dir)

    return all_results


def compare_and_visualize_results(all_results, method_names, current_columns, output_dir):
    """
    比较不同方法的频域特征并可视化

    参数:
    - all_results: 所有方法的频域分析结果列表
    - method_names: 方法名称列表
    - current_columns: 电流列名列表
    - output_dir: 输出目录
    """
    for column in current_columns:
        # 检查该列是否在所有结果中都存在
        if not all(column in result for result in all_results):
            print(f"列 '{column}' 在某些结果中不存在，跳过比较")
            continue

        print(f"\n比较 {column} 的频域特征:")

        # 准备比较数据
        fund_freqs = [result[column]['fundamental_freq'] for result in all_results]
        fund_amps = [result[column]['fundamental_amp'] for result in all_results]
        thds = [result[column]['thd'] for result in all_results]

        # 谐波比较数据
        harmonics = [3, 5, 7, 9, 11]
        harmonic_data = {h: [] for h in harmonics}

        for result in all_results:
            for h in harmonics:
                if h in result[column]['harmonics']:
                    harmonic_data[h].append(result[column]['harmonics'][h])
                else:
                    harmonic_data[h].append(0)

        # 频带能量比较
        band_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
        band_data = {band: [] for band in band_ranges}

        for result in all_results:
            for band in band_ranges:
                if band in result[column]['band_energies']:
                    band_data[band].append(result[column]['band_energies'][band])
                else:
                    band_data[band].append(0)

        # 创建比较图表
        plt.figure(figsize=(16, 12))

        # 1. 基波频率和幅值
        plt.subplot(331)
        plt.bar(method_names, fund_freqs)
        plt.title('基波频率比较')
        plt.ylabel('频率 (Hz)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(332)
        plt.bar(method_names, fund_amps)
        plt.title('基波幅值比较')
        plt.ylabel('幅值')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 2. THD比较
        plt.subplot(333)
        plt.bar(method_names, thds)
        plt.title('总谐波失真比较')
        plt.ylabel('THD (%)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 3. 谐波幅值比较
        plt.subplot(334)
        bar_width = 0.15
        index = np.arange(len(method_names))

        for i, h in enumerate(harmonics):
            plt.bar(index + i * bar_width, harmonic_data[h], bar_width, label=f'{h}次谐波')

        plt.title('谐波幅值比较')
        plt.ylabel('幅值')
        plt.xticks(index + bar_width * 2, method_names, rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 4. 频带能量比较
        plt.subplot(335)
        bar_width = 0.15
        index = np.arange(len(method_names))

        for i, band in enumerate(band_ranges):
            plt.bar(index + i * bar_width, band_data[band], bar_width, label=f'{band[0]}-{band[1]} Hz')

        plt.title('频带能量比较')
        plt.ylabel('能量')
        plt.xticks(index + bar_width * 2, method_names, rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 5. 频谱叠加比较
        plt.subplot(313)
        for i, result in enumerate(all_results):
            plt.semilogy(
                result[column]['spectrum']['freq'],
                result[column]['spectrum']['amp'],
                label=method_names[i]
            )

        plt.title('compare (log)')
        plt.xlabel('fs (Hz)')
        plt.ylabel('x (log)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        filename = os.path.join(output_dir, f'comparison_{column.replace("/", "_")}.png')
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"比较结果已保存到: {filename}")

        # 打印数值比较
        print("\n数值比较:")
        print("方法\t基波频率\t基波幅值\tTHD(%)")
        for i, method in enumerate(method_names):
            print(f"{method}\t{fund_freqs[i]:.2f} Hz\t{fund_amps[i]:.4f}\t{thds[i]:.2f}%")

        print("\n谐波比较:")
        header = "方法"
        for h in harmonics:
            header += f"\t{h}次谐波"
        print(header)

        for i, method in enumerate(method_names):
            line = method
            for h in harmonics:
                line += f"\t{harmonic_data[h][i]:.4f}"
            print(line)

        print("\n频带能量比较:")
        header = "方法"
        for band in band_ranges:
            header += f"\t{band[0]}-{band[1]} Hz"
        print(header)

        for i, method in enumerate(method_names):
            line = method
            for band in band_ranges:
                line += f"\t{band_data[band][i]:.4f}"
            print(line)


def test_interpolation_method(original_data_file, output_dir, sampling_rate, method, current_columns):
    """
    使用指定的插值方法处理数据并返回处理后的文件路径

    参数:
    - original_data_file: 原始数据文件路径
    - output_dir: 输出目录
    - sampling_rate: 采样率
    - method: 插值方法 ('pchip', 'cubic', 'akima', 'savgol', 'linear')
    - current_columns: 电流列名列表

    返回:
    - 处理后的文件路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义输出文件路径
    base_name = os.path.basename(original_data_file).split(".")[0]
    output_file = os.path.join(output_dir, f"{base_name}_{method}.csv")

    # 读取原始数据
    df = pd.read_csv(original_data_file)

    # 创建一份副本
    result_df = df.copy()

    # 对每列进行指定的插值处理
    for col in current_columns:
        if col not in df.columns:
            print(f"警告: 列 '{col}' 在数据文件中不存在，将跳过")
            continue

        # 获取原始数据
        data = df[col].values
        indices = np.arange(len(data))

        # 人为引入一些缺失值来测试插值效果
        # 随机选择10%的点标记为NaN
        np.random.seed(42)  # 固定随机种子以便比较
        missing_indices = np.random.choice(indices, size=int(len(indices) * 0.1), replace=False)
        test_data = data.copy()
        test_data[missing_indices] = np.nan

        # 应用插值方法
        if method == 'pchip':
            # 获取非缺失值的索引和值
            valid_indices = np.where(~np.isnan(test_data))[0]
            valid_values = test_data[valid_indices]
            # 创建插值器
            pchip = PchipInterpolator(valid_indices, valid_values)
            # 插值
            interpolated = test_data.copy()
            interpolated[missing_indices] = pchip(missing_indices)

        elif method == 'cubic':
            # 获取非缺失值的索引和值
            valid_indices = np.where(~np.isnan(test_data))[0]
            valid_values = test_data[valid_indices]
            # 创建插值器
            cs = CubicSpline(valid_indices, valid_values)
            # 插值
            interpolated = test_data.copy()
            interpolated[missing_indices] = cs(missing_indices)

        elif method == 'akima':
            try:
                from scipy.interpolate import Akima1DInterpolator
                # 获取非缺失值的索引和值
                valid_indices = np.where(~np.isnan(test_data))[0]
                valid_values = test_data[valid_indices]
                # 创建插值器
                akima = Akima1DInterpolator(valid_indices, valid_values)
                # 插值
                interpolated = test_data.copy()
                interpolated[missing_indices] = akima(missing_indices)
            except ImportError:
                print("Akima插值不可用，回退到PCHIP")
                valid_indices = np.where(~np.isnan(test_data))[0]
                valid_values = test_data[valid_indices]
                pchip = PchipInterpolator(valid_indices, valid_values)
                interpolated = test_data.copy()
                interpolated[missing_indices] = pchip(missing_indices)

        elif method == 'savgol':
            # 先用线性插值填充缺失值
            temp_data = pd.Series(test_data).interpolate(method='linear').values
            # 然后应用Savitzky-Golay滤波进行平滑
            window_length = min(51, len(temp_data) // 4 * 2 + 1)  # 确保窗口长度为奇数且不超过数据长度的四分之一
            window_length = max(window_length, 5)  # 至少为5
            poly_order = min(3, window_length - 1)  # 多项式阶数
            interpolated = savgol_filter(temp_data, window_length, poly_order)

        elif method == 'linear':
            # 线性插值
            interpolated = pd.Series(test_data).interpolate(method='linear').values

        else:
            print(f"未知的插值方法: {method}，使用线性插值")
            interpolated = pd.Series(test_data).interpolate(method='linear').values

        # 更新结果
        result_df[col] = interpolated

    # 保存处理后的数据
    result_df.to_csv(output_file, index=False)
    print(f"使用 {method} 插值方法处理后的数据已保存到: {output_file}")

    return output_file


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='永磁同步电机电流数据频域特征分析')
    parser.add_argument('--input', '-i', required=True, help='原始电流数据文件路径')
    parser.add_argument('--output_dir', '-o', default='frequency_analysis_results', help='输出目录')
    parser.add_argument('--sampling_rate', '-sr', type=int, default=20000, help='采样率(Hz)')
    parser.add_argument('--columns', '-c', nargs='+', default=['cDAQ1Mod2/ai0', 'cDAQ1Mod2/ai2', 'cDAQ1Mod2/ai3'],
                        help='要分析的电流列名')
    parser.add_argument('--methods', '-m', nargs='+', default=['pchip', 'cubic', 'savgol', 'linear'],
                        help='要测试的插值方法')

    args = parser.parse_args()

    # 创建插值测试目录
    interpolation_dir = os.path.join(args.output_dir, 'interpolated_data')

    # 为每种插值方法处理数据
    interpolated_files = []
    for method in args.methods:
        output_file = test_interpolation_method(
            args.input,
            interpolation_dir,
            args.sampling_rate,
            method,
            args.columns
        )
        interpolated_files.append(output_file)

    # 比较不同插值方法的频域特征
    compare_interpolation_methods(
        args.input,
        interpolated_files,
        args.methods,
        args.sampling_rate,
        args.columns
    )

    print("\n分析完成! 结果保存在:", args.output_dir)


if __name__ == "__main__":
    main()