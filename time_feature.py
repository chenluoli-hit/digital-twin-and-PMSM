import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

plt.rcParams['font.family'] = ['SimHei']
# 定义特征提取函数
def extract_features(data):
    """
    从三相电流数据中提取时域特征

    Args:
        data: 包含三相电流的DataFrame

    Returns:
        包含所有提取特征的一维数组
    """
    features = []

    # 获取三相电流数据
    phase_a = data.iloc[:, 0].values
    phase_b = data.iloc[:, 1].values
    phase_c = data.iloc[:, 2].values

    # 对每相电流分别计算特征
    for phase in [phase_a, phase_b, phase_c]:
        # 1. 均值
        mean = np.mean(phase)

        # 2. 有效值(RMS)
        rms = np.sqrt(np.mean(np.square(phase)))

        # 3. 峭度(Kurtosis)
        kurtosis = stats.kurtosis(phase)

        # 4. 峰值因子(Crest Factor) = 峰值/RMS
        peak = np.max(np.abs(phase))
        crest_factor = peak / rms if rms != 0 else 0

        # 5. 波形因子(Form Factor) = RMS/平均整流值
        rectified_mean = np.mean(np.abs(phase))
        form_factor = rms / rectified_mean if rectified_mean != 0 else 0

        # 添加到特征列表
        features.extend([mean, rms, kurtosis, crest_factor, form_factor])

    return np.array(features)


# 主程序
def main():
    # 设置数据路径
    base_path = "1.0kW/dataset"

    # 存储特征和标签
    X = []
    y = []

    # 遍历所有类别文件夹
    for class_folder in range(5):  # 0, 1, 2, 3, 4
        folder_path = os.path.join(base_path, str(class_folder))
        print(f"处理文件夹 {folder_path}...")

        # 获取文件夹中的所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取特征
            features = extract_features(df)

            # 添加到数据集
            X.append(features)
            y.append(class_folder)

    # 转换为NumPy数组
    X = np.array(X)
    y = np.array(y)

    print(f"特征矩阵形状: {X.shape}")
    print(f"标签数组形状: {y.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建并训练MLP分类器
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu',
                        solver='adam', random_state=43, early_stopping=True, verbose=True)

    mlp.fit(X_train_scaled, y_train)

    # 在测试集上评估模型
    y_pred = mlp.predict(X_test_scaled)

    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')

    # 特征重要性分析（通过特征与标签的相关性）
    feature_names = []
    for phase in ['A', 'B', 'C']:
        for feature in ['Mean', 'RMS', 'Kurtosis', 'Crest Factor', 'Form Factor']:
            feature_names.append(f"{phase}_{feature}")

    # 保存特征重要性
    feature_importance = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    plt.figure(figsize=(12, 8))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('相关性 (绝对值)')
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    # 保存模型和标准化器
    import joblib
    joblib.dump(mlp, 'pmsm_fault_mlp_model.pkl')
    joblib.dump(scaler, 'pmsm_fault_scaler.pkl')

    print("模型和标准化器已保存")

    return X, y, mlp, scaler


# 执行主程序
if __name__ == "__main__":
    X, y, model, scaler = main()