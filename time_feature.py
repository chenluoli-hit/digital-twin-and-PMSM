import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = ['Times New Roman']


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


# 自定义MLP类以记录训练过程中的loss和准确率
class CustomMLP(MLPClassifier):
    def __init__(self, max_epochs=50, **kwargs):
        self.max_epochs = max_epochs
        self.train_loss_curve_ = []  # 每个epoch的训练loss
        self.val_loss_curve_ = []  # 每个epoch的验证loss
        self.train_acc_curve_ = []  # 训练准确率
        self.val_acc_curve_ = []  # 验证准确率
        super().__init__(**kwargs)

    def _compute_loss(self, X, y):
        """计算给定数据的loss值"""
        # 获取当前模型的预测结果（概率分布）
        y_prob = self._forward_pass_fast(X)

        # 转换为独热编码
        y_one_hot = np.zeros((y.shape[0], len(np.unique(y))), dtype=np.float64)
        for i, label in enumerate(y):
            y_one_hot[i, label] = 1

        # 计算损失（交叉熵）
        loss = -np.sum(y_one_hot * np.log(y_prob + 1e-10)) / X.shape[0]
        return loss

    def fit(self, X, y, X_val=None, y_val=None):
        # 重置所有曲线数据
        self.train_loss_curve_ = []
        self.val_loss_curve_ = []
        self.train_acc_curve_ = []
        self.val_acc_curve_ = []

        # 原始loss_curve_将包含所有迭代的loss，但我们不会直接使用它
        self.loss_curve_ = []

        # 训练过程中不需要verbose输出
        original_verbose = self.verbose
        self.verbose = False

        # 保存最佳模型参数
        best_val_loss = np.inf
        best_params = None

        # 禁用warm_start，确保每次训练一个完整的epoch
        original_warm_start = self.warm_start
        self.warm_start = False

        for epoch in range(self.max_epochs):
            # 设置只训练一个epoch
            self.max_iter = 1

            # 训练一个epoch
            super().fit(X, y)

            # 计算训练集loss和准确率
            train_loss = self._compute_loss(X, y)
            self.train_loss_curve_.append(train_loss)

            train_pred = self.predict(X)
            train_acc = accuracy_score(y, train_pred)
            self.train_acc_curve_.append(train_acc)

            # 如果提供了验证集，计算验证集loss和准确率
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                self.val_loss_curve_.append(val_loss)

                val_pred = self.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                self.val_acc_curve_.append(val_acc)

                # 记录最佳模型（基于验证loss）
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = [c.copy() for c in self.coefs_], [i.copy() for i in self.intercepts_]

                # 输出信息
                print(f"Epoch {epoch + 1}/{self.max_epochs}, "
                      f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, "
                      f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
            else:
                # 无验证集时只输出训练信息
                print(f"Epoch {epoch + 1}/{self.max_epochs}, "
                      f"train_loss: {train_loss:.6f}, train_acc: {train_acc:.4f}")

            # 启用warm_start为后续epoch准备
            self.warm_start = True

        # 恢复原始设置
        self.verbose = original_verbose
        self.warm_start = original_warm_start

        # 如果有验证集，并且early_stopping为True，使用最佳参数
        if self.early_stopping and X_val is not None and y_val is not None and best_params is not None:
            self.coefs_, self.intercepts_ = best_params
            print(f"使用验证集选择的最佳模型参数 (val_loss: {best_val_loss:.6f})")

        return self

    def plot_learning_curves(self):
        """绘制训练和验证的loss曲线与准确率曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制loss曲线
        epochs = range(1, len(self.train_loss_curve_) + 1)
        ax1.plot(epochs, self.train_loss_curve_, 'b-', label='train Loss')
        if self.val_loss_curve_:
            ax1.plot(epochs, self.val_loss_curve_, 'r-', label='val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('MLP Loss')
        ax1.grid(True)
        ax1.legend()

        # 绘制准确率曲线
        ax2.plot(epochs, self.train_acc_curve_, 'b-', label='train accuracy')
        if self.val_acc_curve_:
            ax2.plot(epochs, self.val_acc_curve_, 'r-', label='val accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('accuracy')
        ax2.set_title('MLP accuracy')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('mlp_learning_curves.png')
        plt.close()


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

    # 创建并训练自定义MLP分类器，限定50轮训练
    mlp = CustomMLP(
        max_epochs=50,
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        random_state=43,
        early_stopping=True
    )

    # 使用自定义的fit方法训练模型，并传入验证集
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=43, stratify=y_train
    )

    mlp.fit(X_train_subset, y_train_subset, X_val, y_val)

    # 绘制学习曲线
    mlp.plot_learning_curves()

    # 保存训练历史到CSV文件
    history_df = pd.DataFrame({
        'epoch': range(1, len(mlp.loss_curve_)+1),
        'train_loss': mlp.train_loss_curve_,
        'val_loss':mlp.val_loss_curve_,
        'train_accuracy': mlp.train_acc_curve_,
        'val_accuracy': mlp.val_acc_curve_
    })
    history_df.to_csv('mlp_training_history.csv', index=False)
    print("训练历史已保存到 mlp_training_history.csv")

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
    plt.savefig('Time_confusion_matrix.png')
    plt.close()

    # 特征重要性分析（通过特征与标签的相关性）
    feature_names = []
    for phase in ['A', 'B', 'C']:
        for feature in ['Mean', 'RMS', 'Kurtosis', 'Crest Factor', 'Form Factor']:
            feature_names.append(f"{phase}_{feature}")

    # # 保存特征重要性
    # feature_importance = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    # plt.figure(figsize=(12, 8))
    # plt.barh(feature_names, feature_importance)
    # plt.xlabel('相关性 (绝对值)')
    # plt.title('特征重要性')
    # plt.tight_layout()
    # plt.savefig('feature_importance.png')
    # plt.close()

    # 保存模型和标准化器
    import joblib
    joblib.dump(mlp, 'pmsm_fault_mlp_model.pkl')
    joblib.dump(scaler, 'pmsm_fault_scaler.pkl')

    print("模型和标准化器已保存")

    return X, y, mlp, scaler


# 执行主程序
if __name__ == "__main__":
    X, y, model, scaler = main()