import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

plt.rcParams['font.family'] = ['Times New Roman']


# 定义频域特征提取函数
def extract_frequency_features(data, fs=20000, f_fundamental=200):
    """
    从三相电流数据中提取频域特征

    Args:
        data: 包含三相电流的DataFrame
        fs: 采样频率，默认20000Hz
        f_fundamental: 基波频率，默认200Hz

    Returns:
        包含所有提取特征的一维数组
    """
    features = []

    # 获取三相电流数据
    phase_a = data.iloc[:, 0].values
    phase_b = data.iloc[:, 1].values
    phase_c = data.iloc[:, 2].values

    # 计算基波对应的频率索引
    n_points = len(phase_a)
    freq_resolution = fs / n_points  # 频率分辨率
    fundamental_idx = int(round(f_fundamental / freq_resolution))

    # 频率阈值范围（考虑到频率泄漏效应）
    freq_tolerance = 2  # 允许的频率索引偏差

    # 对每相电流进行分析
    for phase_name, phase_data in zip(['A', 'B', 'C'], [phase_a, phase_b, phase_c]):
        # 应用汉宁窗减少频谱泄漏
        windowed_data = phase_data * np.hanning(len(phase_data))

        # 计算FFT
        fft_data = np.fft.rfft(windowed_data)
        fft_magnitude = np.abs(fft_data) / (n_points / 2)  # 幅值归一化

        # 计算功率谱密度
        freq = np.fft.rfftfreq(n_points, d=1 / fs)

        # 寻找基波分量 (在fundamental_idx附近找最大值)
        fund_range = slice(max(0, fundamental_idx - freq_tolerance),
                           min(len(fft_magnitude), fundamental_idx + freq_tolerance + 1))
        fundamental_mag = np.max(fft_magnitude[fund_range])

        # 计算总谐波畸变 (THD)
        # 找出所有谐波 (2nd到10th)
        thd_sum_squared = 0
        for h in range(2, 11):  # 2nd到10th谐波
            h_idx = int(round(h * f_fundamental / freq_resolution))
            h_range = slice(max(0, h_idx - freq_tolerance),
                            min(len(fft_magnitude), h_idx + freq_tolerance + 1))

            if h_range.start < len(fft_magnitude) and h_range.stop <= len(fft_magnitude):
                harmonic_mag = np.max(fft_magnitude[h_range])
                thd_sum_squared += harmonic_mag ** 2

        thd = np.sqrt(thd_sum_squared) / fundamental_mag if fundamental_mag > 0 else 0

        # 将频域特征添加到特征列表
        features.extend([fundamental_mag, thd])

        # 计算频谱峰度和偏度（描述频谱分布形状）
        spectrum_skewness = stats.skew(fft_magnitude)
        spectrum_kurtosis = stats.kurtosis(fft_magnitude)
        features.extend([spectrum_skewness, spectrum_kurtosis])

    return np.array(features)


# 主程序
def main():
    # 设置数据路径
    base_path = "1.0kW/dataset"

    # 存储特征和标签
    X = []
    y = []

    # 遍历所有类别文件夹
    class_folders = [str(i) for i in range(5)]  # 0, 1, 2, 3, 4

    for class_folder in class_folders:
        folder_path = os.path.join(base_path, class_folder)
        print(f"处理文件夹 {folder_path}...")

        # 获取文件夹中的所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取频域特征
            features = extract_frequency_features(df)

            # 添加到数据集
            X.append(features)
            y.append(int(class_folder))

    # 转换为NumPy数组
    X = np.array(X)
    y = np.array(y)

    print(f"特征矩阵形状: {X.shape}")
    print(f"标签数组形状: {y.shape}")

    # 划分训练集、验证集和测试集 (60% 训练, 20% 验证, 20% 测试)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 进一步划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    # 此时训练集占60%，验证集占20%，测试集占20%

    print(f"训练集大小: {len(X_train)} (60%)")
    print(f"验证集大小: {len(X_val)} (20%)")
    print(f"测试集大小: {len(X_test)} (20%)")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 创建MLP分类器
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=50, activation='relu',
                        solver='adam', random_state=42, early_stopping=False, verbose=True)

    # 初始化列表存储每个轮次的训练和验证的准确率和损失
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    # 手动训练MLP，记录每个轮次的准确率和损失
    print("训练MLP模型...")

    # 手动训练并跟踪性能
    mlp.partial_fit(X_train_scaled, y_train, classes=np.unique(y))

    # 记录初始化后的性能
    train_loss_history.append(mlp.loss_)
    train_accuracy_history.append(accuracy_score(y_train, mlp.predict(X_train_scaled)))

    # 计算验证集的损失和准确率
    val_pred = mlp.predict(X_val_scaled)
    val_accuracy_history.append(accuracy_score(y_val, val_pred))

    # 使用与训练集相同的方法计算验证集损失 (需要单独计算，因为MLPClassifier不直接提供预测概率的损失)
    val_proba = mlp.predict_proba(X_val_scaled)
    log_proba = np.log(np.clip(val_proba, 1e-10, None))
    val_loss = -np.sum(log_proba[np.arange(len(y_val)), y_val]) / len(y_val)
    val_loss_history.append(val_loss)

    print(f"初始化 - 训练损失: {train_loss_history[-1]:.4f}, 训练准确率: {train_accuracy_history[-1]:.4f}, "
          f"验证损失: {val_loss_history[-1]:.4f}, 验证准确率: {val_accuracy_history[-1]:.4f}")

    for epoch in range(1, 50):  # 已经初始化一次，再训练99次
        mlp.partial_fit(X_train_scaled, y_train)

        # 训练集性能
        train_loss_history.append(mlp.loss_)
        train_accuracy_history.append(accuracy_score(y_train, mlp.predict(X_train_scaled)))

        # 验证集性能
        val_pred = mlp.predict(X_val_scaled)
        val_accuracy_history.append(accuracy_score(y_val, val_pred))

        # 计算验证集损失
        val_proba = mlp.predict_proba(X_val_scaled)
        log_proba = np.log(np.clip(val_proba, 1e-10, None))
        val_loss = -np.sum(log_proba[np.arange(len(y_val)), y_val]) / len(y_val)
        val_loss_history.append(val_loss)

        print(
            f"轮次 {epoch + 1}/50 - 训练损失: {train_loss_history[-1]:.4f}, 训练准确率: {train_accuracy_history[-1]:.4f}, "
            f"验证损失: {val_loss_history[-1]:.4f}, 验证准确率: {val_accuracy_history[-1]:.4f}")

    # 绘制训练过程
    plt.figure(figsize=(12, 10))

    # plt.subplot(2, 2, 1)
    # plt.plot(range(1, 51), train_loss_history, 'b-')
    # plt.title('训练损失')
    # plt.xlabel('轮次')
    # plt.ylabel('损失')
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(range(1, 51), val_loss_history, 'g-')
    # plt.title('验证损失')
    # plt.xlabel('轮次')
    # plt.ylabel('损失')
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(range(1, 51), train_accuracy_history, 'r-')
    # plt.title('训练准确率')
    # plt.xlabel('轮次')
    # plt.ylabel('准确率')
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(range(1, 51), val_accuracy_history, 'm-')
    # plt.title('验证准确率')
    # plt.xlabel('轮次')
    # plt.ylabel('准确率')
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 在测试集上评估模型
    y_pred = mlp.predict(X_test_scaled)

    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")

    # 绘制混淆矩阵
    mpl.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predict class')
    plt.ylabel('Ture class')
    #plt.title('混淆矩阵')
    plt.savefig('confusion_matrix_simplified_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 特征重要性分析（通过特征与标签的相关性）
    feature_names = []
    for phase in ['A', 'B', 'C']:
        feature_names.extend([
            f"{phase}_Fundamental_Magnitude",
            f"{phase}_THD",
            f"{phase}_Spectrum_Skewness",
            f"{phase}_Spectrum_Kurtosis"
        ])

    # 保存特征重要性
    feature_importance = np.abs(np.corrcoef(X.T, y)[:-1, -1])

    # plt.figure(figsize=(14, 10))
    # sorted_idx = np.argsort(feature_importance)
    # plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
    # plt.xlabel('相关性 (绝对值)')
    # plt.title('频域特征重要性')
    # plt.tight_layout()
    # plt.savefig('simplified_frequency_feature_importance.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # 保存模型和标准化器
    import joblib
    joblib.dump(mlp, 'pmsm_fault_simplified_frequency_mlp_model.pkl')
    joblib.dump(scaler, 'pmsm_fault_simplified_frequency_scaler.pkl')

    # 保存训练历史数据
    history_df = pd.DataFrame({
        'epoch': range(1, 51),
        'train_loss': train_loss_history,
        'train_accuracy': train_accuracy_history,
        'val_loss': val_loss_history,
        'val_accuracy': val_accuracy_history
    })
    history_df.to_csv('training_history.csv', index=False)

    print("模型、标准化器和训练历史已保存")

    return X, y, mlp, scaler, history_df


# 执行主程序
if __name__ == "__main__":
    X, y, model, scaler, history = main()