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
# lec11_JAK1.csv を読み込み、dataというオブジェクトとしなさい
""" WRITE HERE """
data = pd.read_csv(r".\lec11_JAK1.csv")

# 説明変数をXとしなさい
# 修正の必要はない
X = data[data.columns[3:]]

# PCA()をpca_mdlとしなさい
""" WRITE HERE """
pca_mdl = PCA()

# pca_mdlにfitメソッドを用いて X で学習させなさい
# 標準化を行わないことに注意すること
""" WRITE HERE """
pca_mdl.fit(X)

# 学習したpca_mdlのtransformメソッドを用いて X を変換しなさい
# この時、変換したデータはpca_corに格納しなさい
""" WRITE HERE """
pca_cor = pca_mdl.transform(X)

### 以降は修正不要 ###
acc_explained_variance_ratio = [sum(pca_mdl.explained_variance_ratio_[:i]) for i in range(len(pca_mdl.explained_variance_ratio_))]

# グラフの表示
plt.xlim(1,1800)
plt.ylim(0,1.05)
plt.xlabel('# PCA Component')
plt.ylabel('Explained Variance Ratio of PCA')
plt.title('Explained Variance Ratio of PCA')
plt.plot(range(len(pca_mdl.explained_variance_ratio_)), acc_explained_variance_ratio)
plt.savefig('lec11_Exr_PCA_1.png', dpi=300, bbox_inches='tight')
plt.close()
print("参考資料 lec11_Exr_PCA_1.png を出力しました")


plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.scatter(pca_cor[:, 0], pca_cor[:, 1], edgecolor='k')
plt.title('2D PCA Scatter Plot of Molecules')
plt.savefig('lec11_kadai_PCA_2D.png', dpi=300, bbox_inches='tight')
print("lect1_kadai_PCA_2D.png を出力しました")

# 3Dプロットの準備
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 散布図のプロット
sccat = ax.scatter(pca_cor[:, 0], pca_cor[:, 1], pca_cor[:, 2])

# 軸ラベルとタイトル
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D PCA Scatter Plot of Molecules')
plt.savefig('lec11_kadai_PCA_3D.png', dpi=300, bbox_inches='tight')
plt.close()
print("提出用ファイル lec11_kadai_PCA_3D.png を出力しました")

# 来週、使用するファイル
c_name = ['PC'+str(i) for i in range(len(pca_mdl.explained_variance_ratio_))]
df_pca = pd.DataFrame(pca_cor,columns=c_name)
df_pca.to_csv('lec11_pca.csv',index=False)
print('lec11_pca.csv を出力しました')
