from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier  # 导入DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import expit

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 加载数据集
file_path = 'data/405case-no nickson.xlsx'
data = pd.read_excel(file_path)
label_encoder = LabelEncoder()

# 准备数据
X = data[['HR', 'N']]
y = data['Stability']
y_encoded = label_encoder.fit_transform(y)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=16)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建回归模型
model1 = GradientBoostingClassifier(
    n_estimators=33,
    learning_rate=0.0664961169465023,
    max_depth=4,
    min_samples_split=6,
    min_samples_leaf=4,
    subsample=0.7727548367453908,
    random_state=1
)

model2 = CatBoostClassifier(
    learning_rate=0.4959728569177294,
    depth=8,
    l2_leaf_reg=2.457755483883723,
    iterations=25,
    verbose=0,
    random_state=1
)


model3 = RandomForestClassifier(
    n_estimators=46,
    max_depth=6,
    min_samples_split=14,
    min_samples_leaf=1,
    max_features='log2',
    criterion='entropy',
    random_state=1
)

base_models = [
    ('gbdt', model1),
    ('catboost', model2),
    ('rf', model3),
    #('knn', model4),
    #('catboost', model5),
]

# best_params = study.best_params
final_estimator = LogisticRegression(
    C=3.3548931846470196,
    penalty='l2',
    solver='saga',
    max_iter=500,
    random_state=1
)

# 使用最优超参数训练决策树模型
best_model = StackingClassifier(estimators=base_models, final_estimator=final_estimator, passthrough=True)

# 训练模型
best_model.fit(X_train_scaled, y_train)

# 获取决策边界的范围
x_min, x_max = 1e0, X['HR'].max() + 0.5
y_min, y_max = 1e-1, 1e3


# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 3000))
grid = np.c_[xx.ravel(), yy.ravel()]

# 将网格点标准化
grid_scaled = scaler.transform(grid)

# 获取预测结果并绘制决策边界
Z = best_model.predict(grid_scaled)
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 6))
# 计算直线方程的斜率和截距
x_start, y_start = 1, 10 ** (-685.782 / 1060.4725)
x_end, y_end = 10 ** (2 * 3177.056 / 4298.2766), 10 ** (3 * 2385.4088 / 3180.2008)

# 限制x_end为0.55e2，并计算对应的y_end
x_limit = X['HR'].max() + 0.5
y_limit = y_start + (y_end - y_start) * (x_limit - x_start) / (x_end - x_start)

# 绘制限制后的直线
plt.plot([x_start, x_limit], [y_start, y_limit], color='white', linestyle='--', linewidth=4)

plt.contourf(xx, yy, Z, alpha=0.5, cmap='RdYlBu')
# 绘制决策边界的轮廓线，并加粗
contour = plt.contour(xx, yy, Z, colors='k', levels=[0.5], linewidths=3)

# 绘制原始数据的散点图
plt.scatter(X['HR'], X['N'], c=y_encoded, edgecolors='k', cmap='RdYlBu')

# 设置对数坐标轴
plt.xscale('log')
plt.yscale('log')

# 自定义刻度
plt.xticks([1e0, 1e1, 0.32e2], labels=['$10^0$', '$10^1$', '$0.32 \\times 10^2$'], fontsize=16, fontname='Times New Roman')
plt.yticks([1e-1, 1e0, 1e1, 1e2, 1e3], labels=['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$'], fontsize=16, fontname='Times New Roman')
# 通过 tick_params 设置刻度字体

# 手动设置每个刻度的字体
ax = plt.gca()  # 获取当前轴
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(18)
# plt.ylim(1e-1, 1e3)
plt.xlabel('Hydraulic Radius (m)', size=20)
plt.ylabel('Stability Number, N', size=20)
# 设置坐标刻度朝内
plt.tick_params(axis='both', which='both', direction='in', length=6)  # axis='both' 设置 X 和 Y 轴，length 设置刻度线的长度



plt.colorbar()
plt.savefig('plot_finished_0714/决策边界-stacking 405case.tif', dpi=600, format='tiff')
plt.show()

