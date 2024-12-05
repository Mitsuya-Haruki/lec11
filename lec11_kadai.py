# 3次元のグラフを書くときに必要なインポート
from mpl_toolkits.mplot3d import Axes3D

# pandasをpdとしてインポートしなさい
""" WRITE HERE """
import pandas as pd #l

# numpyをnpとしてインポートしなさい
""" WRITE HERE """
import numpy as np #l

# matplotlib.pyplotをpltとしてインポートしなさい
""" WRITE HERE """
import matplotlib.pyplot as plt

# sklearn.decomposition から PCA をインポートしなさい
# from XXXX import YYYY の形になる
""" WRITE HERE """
from sklearn.decomposition import PCA

# pandasのread_csvメソッドを利用して
# lect1_2A1.csv を読み込み、dataというオブジェクトとしなさい
""" WRITE HERE """
data = pd.read_csv(r".\lec11_JAK1.csv")

# 説明変数Xをとり出し、修正版としなさい
X = data[data.columns[3:]]

# PCA()をpca_mdlとしなさい
""" WRITE HERE """
pca_mdl = PCA()

# pca_mdlにfitメソッドでXを学習させなさい
""" WRITE HERE """
pca_mdl.fit(X)

# pca_mdlのtransformメソッドでXを変換して
# その結果をpca_corに格納しなさい
""" WRITE HERE """
pca_cor = pca_mdl.transform(X)

### 結果の描画 ###
acc_explained_variance_ratio = [sum(pca_mdl.explained_variance_ratio_[:i]) for i in range(len(pca_mdl.explained_variance_ratio_))]

plt.figure()
plt.plot(range(len(pca_mdl.explained_variance_ratio_)), acc_explained_variance_ratio)
plt.xlabel('PCA Component')
plt.ylabel('Explained Variance Ratio of PCA')
plt.title('Explained Variance Ratio of PCA')
plt.savefig('lect1_kadai_PCA_1.png', dpi=300, bbox_inches='tight')
plt.close()
print("lect1_kadai_PCA_1.png を出力しました")

plt.figure()
plt.scatter(pca_cor[:, 0], pca_cor[:, 1], edgecolor='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA Scatter Plot of Molecules')
plt.savefig('lect1_kadai_PCA_2D.png', dpi=300, bbox_inches='tight')
plt.close()
print("lect1_kadai_PCA_2D.png を出力しました")

# 3Dプロットの準備
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 散布図のプロット
scat = ax.scatter(pca_cor[:, 0], pca_cor[:, 1], pca_cor[:, 2])
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.title('3D PCA Scatter Plot of Molecules')
plt.savefig('lect1_kadai_PCA_3D.png', dpi=300, bbox_inches='tight')
plt.close()
print("lect1_kadai_PCA_3D.png を出力しました")

# 累積寄与率の表示
for i in range(len(pca_mdl.explained_variance_ratio_)):
    print(f"累積寄与率（第{i+1}主成分まで）：{acc_explained_variance_ratio[i]}")
