# はじめに
世の中には時系列データがたくさんあって、例えばサイトの日々のアクセス数だとか、工場の温度センサーだったりとか、いろいろな現場で目に付く機会が多いと思う。
そんな時系列データを使った課題解決の中でもニーズが高いのが異常検知だと思われる。
異常検知へのアプローチはたくさんあるが、その中の一つMT（マハラノビス・タグチ）法を遊びで試してみたので書く。

# MT法
MT法の基本的な概念やRでの実装方法は井手剛さんの書籍[「入門 機械学習による異常検知 Rによる実践ガイド」](https://www.amazon.co.jp/dp/4339024910)に書いてある。
マハラノビス距離に基づく外れ値検出手法に、異常変数の選択手法を組み合わせることで、多変数のホテリング理論の課題に対応した手法である。
イメージ的に、相関のある複数のデータ同士をお互いに比較しておかしくないのか見ている感じ。
（AとBとCは同じような動きをするはずなのに、Aだけ異なる動きをしていた ⇒ 異常と判断 ⇒ MT法は異常を起こしているデータの判別も可能）
※ユークリッド距離とは違い、マハラノビス距離はデータ同士の相関関係を考慮した距離である
※マハラノビス距離に基づく外れ値検出手法イメージ図
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/99a40622-8e00-5db7-0381-bae744e6042f.png)

MT法の流れは以下の4ステップである。

1. 正常データをもとに標本平均と標本共分散を求める
2. 正常データの各標本に対し、1変数当たりのマハラノビス距離(=異常度とする)を計算する
3. 正常データ内で計算したマハラノビス距離をもとに閾値を決める
4. 正常データから求めた標本平均と標本共分散を使って異常を含むデータのマハラノビス距離を計算する

ステップ4で計算したマハラノビス距離がステップ3で決めた閾値を超えた場合、異常とみなす。
また、SN比という指標を使って、個々の変数の寄与が数値化され、どの変数によって異常が起きているのかを判別することが可能である。
$$ SN_q = -10\log_{10}\biggl(\frac{1}{N^{'}}\sum_{n=1}^{N^{'}}\frac{1}{a_{q}(x^{'(n)})/M_q} \biggr)$$
$M$は変数の数、$N^{'}$は異常データの数、$a$は異常度、$ q $は変数の取捨選択パターンを区別する添字であり、1変数ずつSN比を見る場合$q$は$M$通りあり、$ M_q=1 $となり、ある1つの異常時点($N^{'}=1$)のSN比の式は簡単になる。

```math
\begin{align}
SN_q&=-10\log_{10}\biggl(\frac{1}{N^{'}}\sum_{n=1}^{N^{'}}\frac{1}{a_{q}(x^{'(n)})/M_q} \biggr)\\
&=10\log_{10}\biggl(\frac{a_{q}(x^{'})}{M_q}\biggr)\\
&=10\log_{10}\biggl(\frac{(x_{q}^{'}-\hat{\mu}_{q})^2}{\hat{\sigma}_q^{2}}\biggr)
\end{align}
```
今回は1変数ずつSN比を見るのでこの簡単に示したSN比を使う。
式から、平均からの偏差が大きいほどSN比が大きくなることがわかる。つまりSN比が大きいほどその変数の寄与が高いとみなせる。

# データ
都合の良いデータが見当たらなかったので、適当にA,B,C,Dの4種のデータを作成した。
データAのindex番号800番目に異常値を挿入した。

```{python}
# パッケージインポート
import os
import re
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import scipy
import functools
import seaborn as sns
import sys
import tqdm as tq
from tqdm import tqdm
import gzip
import glob
import datetime as dt
import gc
import time
sns.set()

# sin波にランダムにノイズを乗せたデータ4種を生成
x=np.linspace(1,500,1001)
arr = np.random.normal(-0.3, 0.3, len(x))# 乱数
data1=np.sin(x)+arr
arr2 = np.random.normal(-0.16, 0.16, len(x))# 乱数
data2=np.sin(x)+arr2
arr3 = np.random.normal(-0.11, 0.11, len(x))# 乱数
data3=np.sin(x)+arr3
arr4 = np.random.normal(-0.12, 0.12, len(x))# 乱数
data4=np.sin(x)+arr4
data1[800]=2
data=pd.DataFrame(np.hstack((data1.reshape(-1,1),data2.reshape(-1,1),data3.reshape(-1,1),data4.reshape(-1,1))), columns=['A','B','C','D'])
display(data)

colors = list(mpl.cm.Set1.colors)+list(mpl.cm.Set2.colors)+list(mpl.cm.Set3.colors)
fig=plt.figure(figsize=(15,8))
plt.plot(range(len(x)),data1,label='A',alpha=0.9, c=colors[1])
plt.plot(range(len(x)),data2,label='B',alpha=0.6, c=colors[2])
plt.plot(range(len(x)),data3,label='C',alpha=0.6, c=colors[3])
plt.plot(range(len(x)),data4,label='D',alpha=0.6, c=colors[4])
plt.legend()
plt.show()

# A,B,C,Dの相関関係
display(data[['A','B','C','D']].corr())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ae668c8a-e59d-b72f-b763-4f6c6b2111a1.png)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/a6e766e2-6291-222b-bf23-afe6a296ead1.png)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/da8bff86-087f-0589-4f6a-38e6f0f96b0b.png)


データAのindex番号800番目の異常を検知することを目指す。

# 正常データを使用した計算
Index番号が600までの正常データから、標本平均と標本共分散を求め、1変数当たりのマハラノビス距離を計算する。
ここで求めることができた標本平均と標本共分散を、後ほどの異常を含むデータのマハラノビス距離の計算にも使用する。

```{python}
# 1変数当たりのマハラノビス距離計算
def mahalanobis_list(data):
    # "data" is DataFrame    
    two_data=data.astype(float).values
    two_data_col=data.columns.values
    # 平均
    mean = np.mean(two_data, axis=0)
    # データから平均を引いた値
    two_data_m = two_data - mean
    # 分散共分散行列を計算
    cov = np.cov(two_data.T)
    # 分散共分散行列の逆行列を計算
    cov_i = np.linalg.pinv(cov)

    # distance.mahalanobis
    # データ: two_data[i], 平均値: mean, 共分散行列の逆行列: np.linalg.pinv(cov) から距離を計算
    print('Calculation of Mahalanobis Distance')
    m_d=[(scipy.spatial.distance.mahalanobis(two_data[i], mean, cov_i))/len(two_data_col) for i in tqdm(range(len(two_data)))]
    
    # SN比解析
    print('Calculation of SN Ratio')
    sn=[]
    for i,row in tqdm(enumerate(two_data_m)):
        xc_prime=two_data_m[i,:]
        SN1=10*np.log10((xc_prime**2)/np.diag(cov))
        sn.append(SN1)
        
    return np.array(m_d), mean, cov, cov_i, np.array(sn), two_data_col

# Index番号が600までの正常データ
m_d, mean, cov, cov_i, SN1, two_data_col = mahalanobis_list(data.iloc[:600,:])
print('MD shape',m_d.shape)
print('A,B,C,D mean:',mean)
print('A,B,C,D cov:\n',cov)
print('SN ratio shape',SN1.shape)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e5507630-c5ae-bece-78cf-2c423818fbca.png)

正常データと、正常データのマハラノビス距離をplotしてみる。
マハラノビス距離（MD）は正常データではおおよそ1以下になっていることが確認できる。

```{python}
# マハラノビス距離を可視化
colors = list(mpl.cm.Set1.colors)+list(mpl.cm.Set2.colors)+list(mpl.cm.Set3.colors)
plt.figure(figsize=(15,4))
ax=plt.subplot(1,1,1)
ax2=ax.twinx()
train_mean=[]
train_cov=[]
train_cov_i=[]
train_mean.append(mean)
train_cov.append(cov)
train_cov_i.append(cov_i)
ax.plot(data.iloc[:600,:].index, m_d, c=colors[0], alpha=0.8, label='MD')
for j, col_k in enumerate(data.iloc[:600,:].columns.values):
    # plot用に正規化
    mm=MinMaxScaler()
    ax2.plot(data.iloc[:600,:].index, mm.fit_transform(data.iloc[:600,:][col_k].values.reshape(-1,1)), c=colors[j+1], alpha=0.6, label=col_k)
ax.set_ylim(-0.1,4)
ax2.set_ylim(-2,1.1)
ax2.grid(False)
ax.set_xlabel('Index')
ax.set_ylabel('MD')
ax2.set_ylabel('data')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='center left')
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/595f7a2e-f453-6bad-22e4-8b72069cd5e4.png)

# 異常データにMT法を適用
正常データで求めた標本平均と標本共分散を使って、Index番号600以降の異常を含むデータに対して、マハラノビス距離を求める。

```{python}
# すでに計算した平均や共分散を使ってマハラノビス距離を計算
def mahalanobis_apply(data,mean,cov,cov_i):# mean,cov,cov_iは正常データで求めたもの
    # "data" is DataFrame
    two_data=data.astype(float).values
    two_data_col=data.columns.values
    # データから平均引いた値
    two_data_m = two_data - mean

    # distance.mahalanobis
    # データ: two_data[i], 平均値: mean, 共分散行列の逆行列: np.linalg.pinv(cov) から距離を計算
    print('Calculation of Mahalanobis Distance')
    m_d=[(scipy.spatial.distance.mahalanobis(two_data[i], mean, cov_i))/len(two_data_col) for i in tqdm(range(len(two_data)))]
    
    # SN比解析
    print('Calculation of SN Ratio')
    sn=[]
    for i,row in tqdm(enumerate(two_data_m)):
        xc_prime=two_data_m[i,:]
        SN1=10*np.log10((xc_prime**2)/np.diag(cov))
        sn.append(SN1)
        
    return np.array(m_d), np.array(sn), two_data_col

m_d2, SN2, two_data_col2 = mahalanobis_apply(data.iloc[600:,:],train_mean[0],train_cov[0],train_cov_i[0])
print('MD shape',m_d2.shape)
print('SN ratio shape',SN2.shape)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/33510292-c473-181f-2c94-29fd7af8f1a7.png)

求めたマハラノビス距離をplotしてみる。
Index番号800番目のマハラノビス距離が2.5程度と他よりも大きくなっていることが確認できる。

```{python}
# すでに計算した平均や共分散を使って計算したマハラノビス距離を可視化
plt.figure(figsize=(15,4))
ax=plt.subplot(1,1,1)
ax2=ax.twinx()
ax.plot(data.iloc[600:,:].index, m_d2, c=colors[0], alpha=0.8, label='MD')
for j, col_k in enumerate(data.iloc[:600,:].columns.values):
    # plot用に正規化
    mm=MinMaxScaler()
    ax2.plot(data.iloc[600:,:].index, mm.fit_transform(data.iloc[600:,:][col_k].values.reshape(-1,1)), c=colors[j+1], alpha=0.6, label=col_k)
    
ax.set_ylim(-0.1,4)
ax2.set_ylim(-2,1.1)
ax.set_xlabel('Index')
ax.set_ylabel('MD')
ax2.set_ylabel('data')
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='center left')
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ea02d6f7-b0cb-bda5-d08f-472efb062548.png)

正常データと異常データを繋げてもう一度可視化してみる。
青色部分は正常として標本平均や標本共分散を求めたデータ。
橙色部分は異常を含むデータである。

```{python}
# データを結合
m_d_all=np.hstack((m_d,m_d2))# マハラノビス距離
SN_all=np.vstack((SN1,SN2))# SN比

plt.figure(figsize=(15,4))
ax=plt.subplot(1,1,1)
ax2=ax.twinx()
ax.plot(data.index, m_d_all, c=colors[0], alpha=0.8, label='MD')
for j, col_k in enumerate(data.columns.values):
    # plot用に正規化
    mm=MinMaxScaler()
    ax2.plot(data.index, mm.fit_transform(data[col_k].values.reshape(-1,1)), c=colors[j+1], alpha=0.6, label=col_k)
    
ax.set_ylim(-0.1,4)
ax2.set_ylim(-2,1.1)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='center left')
ax.fill_between(data.index[:600], -0.1, 4, color="c", alpha=0.3)
ax.fill_between(data.index[600:], -0.1, 4, color="y", alpha=0.3)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/15d1125a-2775-eb38-24f6-3539e4d80d39.png)

SN比もすでに求めているので、Index番号800番時点の、変数A,B,C,DのSN比を見てみる。
plotした結果、AのSN比が最も大きく、異常にはAが大きく寄与したと言える。
実際にデータAのIndex番号800に異常値を入れたので、これは正解である。

```{python}
# SN比の可視化
def plot_sn_ratio(two_data_col, inds, SN1, index, figsize=(10,10)):
    plt.figure(figsize=figsize)
    for i, ind in enumerate(inds):
        ax=plt.subplot(int(np.ceil(np.sqrt(len(inds)))),int(np.ceil(np.sqrt(len(inds)))),i+1)
        sns.barplot([two_data_col[i].split('_')[-1] for i in range(len(two_data_col))],SN1[ind,:], ax=ax)
        ax.set_title('Index:'+str(int(index[ind]))+',  Annomaly data name is '+two_data_col[np.argmax(SN1[ind,:])])
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

inds=[800] #inds番目に異常が起きていたindex番号を入れる
plot_sn_ratio(data.columns.values, inds, SN_all, data.index, figsize=(10,5))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/dcdaac89-f116-5d8e-6c01-afe5c54ea01d.png)

# おわりに
MT法を試してみた。実際の時系列データで試したわけではないので実用性を判断することはできないが、互いに相関のあるデータが複数あって異常検知するなら使えるような気がする。
基本的に井手さんの書籍[「入門 機械学習による異常検知 Rによる実践ガイド」](https://www.amazon.co.jp/dp/4339024910)内のRコードを参考にしたので、気になった人は書籍を見ることをオススメする。（名著）
以上！

# 追記
マハラノビス距離が実際にデータに対してどのようになっているか、2変数で可視化してみる。
まずさっきのように正常データで標本平均や標本共分散を求める。

```{python}
# Index番号が600までの正常データ
m_d, mean, cov, cov_i, SN1, two_data_col = mahalanobis_list(data.iloc[:600,:2])
train_mean=[]
train_cov=[]
train_cov_i=[]
train_mean.append(mean)
train_cov.append(cov)
train_cov_i.append(cov_i)

# Index番号が600以降の異常データ
m_d2, SN2, two_data_col2 = mahalanobis_apply(data.iloc[600:,:2],train_mean[0],train_cov[0],train_cov_i[0])
```

いったん可視化。

```{python}
# データを結合
m_d_all=np.hstack((m_d,m_d2))# マハラノビス距離
SN_all=np.vstack((SN1,SN2))# SN比

plt.figure(figsize=(15,4))
ax=plt.subplot(1,1,1)
ax2=ax.twinx()
ax.plot(data.index, m_d_all, c=colors[0], alpha=0.8, label='MD')
for j, col_k in enumerate(data.columns.values[:2]):
    # plot用に正規化
    mm=MinMaxScaler()
    ax2.plot(data.index, mm.fit_transform(data[col_k].values.reshape(-1,1)), c=colors[j+1], alpha=0.6, label=col_k)

ax.set_ylim(-0.1,4)
ax2.set_ylim(-2,1.1)
ax2.grid(False)
ax.set_xlabel('Index')
ax.set_ylabel('MD')
ax2.set_ylabel('data')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='center left')
ax.fill_between(data.index[:600], -0.1, 4, color="c", alpha=0.3)
ax.fill_between(data.index[600:], -0.1, 4, color="y", alpha=0.3)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7a075098-47eb-7799-4fd2-ac1841cd3fef.png)

正常時と異常時のデータそれぞれの散布図の上にマハラノビス距離の等高線を引く。
マハラノビス距離は座標左下から右上に長くなっていて、異常時のデータは明らかにマハラノビス距離でいうと遠い。
このように相関のあるデータの場合、マハラノビス距離を使用することが効果的であることがわかると思う。

```{python}
# 2変数の取り得る値とマハラノビス距離の対応を配列として定義
# 今回はデータが-4から4まで変化した時のそれぞれのマハラノビス距離を計算している。
mah_dis=[]
for i in np.round(np.linspace(-4, 4, 18)):
    mah_dis_one=[]
    for j in np.round(np.linspace(-4, 4, 18)):
        mah_dis_one.append((i, j, scipy.spatial.distance.mahalanobis([i,j], mean, cov_i)))
    mah_dis.append(mah_dis_one)
mah_dis=np.array(mah_dis)

# グラフ描画
fig=plt.figure(figsize=(12,6))
colors = list(mpl.cm.Set1.colors)+list(mpl.cm.Set2.colors)+list(mpl.cm.Set3.colors)

ax=plt.subplot(1,2,1)
# 等高線
contour = ax.contour(mah_dis.transpose()[0], mah_dis.transpose()[1], mah_dis.transpose()[2]
                     , levels=np.linspace(-4, 4, 18), colors=['k'],alpha=0.4)
contour.clabel(fmt='%1.1f', fontsize=10)
# 散布図
ax.plot(data.iloc[:600,0].values
        ,data.iloc[:600,1].values
        , '.', color=colors[2], alpha=0.5)
ax.set_xlabel(data.columns[0])
ax.set_ylabel(data.columns[1])
ax.set_title('Normal\nMax Mahalanobis Distance: '+str(np.round(max(m_d),2)))

ax2=plt.subplot(1,2,2)
# 等高線
contour = ax2.contour(mah_dis.transpose()[0], mah_dis.transpose()[1], mah_dis.transpose()[2]
                      , levels=np.linspace(-4, 4, 18), colors=['k'],alpha=0.4)
contour.clabel(fmt='%1.1f', fontsize=10)
# 散布図
ax2.plot(data.iloc[600:,0].values
        ,data.iloc[600:,1].values
        , '.', color=colors[1], alpha=0.5)
ax2.set_xlabel(data.columns[0])
ax2.set_ylabel(data.columns[1])
ax2.set_title('Abnormal\nMax Mahalanobis Distance: '+str(np.round(max(m_d2),2)))
anom_x=data.iloc[600:,0].values[np.where(m_d2==max(m_d2))[0][0]]
anom_y=data.iloc[600:,1].values[np.where(m_d2==max(m_d2))[0][0]]
ax2.scatter(anom_x
            ,anom_y
            , color=colors[0])
ax2.text(anom_x, anom_y, '  abnormal data', ha='left', fontsize=9)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/47295b8f-d74f-ec4b-c348-1be4d2936c07.png)

これを応用して、例えば機械学習の推論値と実測値の2変数のマハラノビス距離を計算して異常検知なども効果的だと思われる。
