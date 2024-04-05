# 目次
[はじめに](#はじめに)
[参考](#参考)
[使用パッケージimport](#使用パッケージimport)
[データ説明&読み込み](#データ説明読み込み)
[前処理](#前処理)
[想定される因果(実際は不明)](#想定される因果実際は不明)
[軽くデータ俯瞰](#軽くデータ俯瞰)
[LiNGAM関連](#lingam関連)
　[DirectLiNGAM](#directlingam)
　[DirectLiNGAM-bootstrap](#directlingam-bootstrap)
　[RESIT](#resit)
　[RESIT-bootstrap](#resit-bootstrap)
　[RCD](#rcd)
　[RCD-bootstrap](#rcd-bootstrap)
　[CAM-UV](#cam-uv)
　[LiM](#lim)
[NOTEARS](#notears)
[Bayesian network](#bayesian-network)
[Graphical Lasso](#graphical-lasso)
[考察](#考察)  
[おわりに](#おわりに)

# はじめに
因果探索手法について、各手法で実際に結果にどのような違いが出るのかざっくり比較するためにUCIの「AI4I 2020 Predictive Maintenance Dataset Data Set」を使ってDAGを推定してみた。
各手法について、きれいにまとめて理論がどうのこうの書いているわけではなく、説明は公式ページなどの文章をDeepLで翻訳して貼り付けたりしているだけ。本題はライブラリ使ってDAGを推定して比較すること。
理論とかちゃんと知りたい人は公式ページとか論文とか他の人の記事を見ていただければ、と。
自分のために最近作ったJupyter Notebookの内容をそのまま記事にしたのでわかりにくいところはあると思う。

# 参考
[UCIサイトページ](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
[lingamのgithub](https://github.com/cdt15/lingam)
[博士課程進学率に関する因果モデルの構築 -統計的因果探索アルゴリズム"LiNGAM"による試行的分析-](https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/1)
[lingamのtutorial](https://lingam.readthedocs.io/en/latest/tutorial/lingam.html)
[Just-In-Timeモデルを利用した非定常非線形時系列データに対する因果探索](https://www.jstage.jst.go.jp/article/pjsai/JSAI2022/0/JSAI2022_3E4GS205/_pdf/-char/ja)
[bootstrapのtutorial](https://lingam.readthedocs.io/en/latest/tutorial/bootstrap.html)
[resitのtutorial](https://lingam.readthedocs.io/en/latest/tutorial/resit.html)
[rcdのtutorial](https://lingam.readthedocs.io/en/latest/tutorial/rcd.html)
[camuvのtutorial](https://lingam.readthedocs.io/en/latest/tutorial/camuv.html)
[limのtutorial](https://lingam.readthedocs.io/en/latest/tutorial/lim.html)
[causalnex公式](https://causalnex.readthedocs.io/en/latest/)
[causalnexのgithub](https://github.com/quantumblacklabs/causalnex/)
[日本語記事:PythonでNOTEARS・ベイジアンネットによる因果グラフ推定 -causalnexの紹介-](https://yuminaga.hatenablog.com/entry/2021/05/23/135852)
[日本語記事:Pythonによる因果グラフ推定 -causalnexの紹介 その2-](https://yuminaga.hatenablog.com/entry/2021/05/23/170842)
[日本語記事:因果探索ライブラリcausalnex](https://socinuit.hatenablog.com/entry/2021/03/27/224414)
[論文「DAGs with NO TEARS:Continuous Optimization for Structure Learning」](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)  
[Qiita記事「DAG の構造学習を連続最適化問題に落とし込んで解くNO TEARSアルゴリズム」](https://qiita.com/kueda_cs/items/5b163bd778abe1b109e8)
[pgmpy公式](https://pgmpy.org/index.html)  
[What are Bayesian Models](https://pgmpy.org/detailed_notebooks/2.%20Bayesian%20Networks.html#)  
[Pythonによる因果分析](https://www.amazon.co.jp/dp/4839973571)
[Learning Bayesian Networks from Data](https://pgmpy.org/detailed_notebooks/10.%20Learning%20Bayesian%20Networks%20from%20Data.html)
[PCアルゴリズム](https://speakerdeck.com/s1ok69oo/pcarugorizumuniyorubeiziannetutowaku)
["MmhcEstimator()"](https://pgmpy.org/structure_estimator/mmhc.html)
["MmhcEstimator()"](https://pgmpy.org/structure_estimator/mmhc.html)
["HillClimbSearch()"](https://pgmpy.org/structure_estimator/hill.html)
[scikit-learn: 2.6.3. Sparse inverse covariance](https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance)

# 使用パッケージimport
## いろいろimport(必要ないものもある)
```python
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.font_manager as fm
import matplotlib.font_manager as font_manager
from matplotlib.markers import TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN, CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import scipy
import functools
import seaborn as sns
import glob
import datetime as dt
import gc
import sys
import tqdm as tq
from tqdm import tqdm
import time
import pickle
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import collections
jpn_fonts=list(np.sort([ttf for ttf in fm.findSystemFonts() if 'ipaexg' in ttf or 'msgothic' in ttf or 'japan' in ttf or 'ipafont' in ttf]))
jpn_font=jpn_fonts[0]
prop = font_manager.FontProperties(fname=jpn_font)
print(jpn_font)
sns.set()

import warnings
warnings.simplefilter('ignore')
```

## LiNGAM import
```python
import graphviz
import lingam
from lingam.utils import make_dot
from lingam.utils import print_causal_directions, print_dagc
```

## CausalNex import
```python
# causalnexはNOTEARSが使える
import causalnex
from causalnex.structure.notears import from_pandas
from causalnex.structure.notears import from_pandas_lasso
from causalnex.structure.pytorch import from_pandas as from_pandas_pytorch
from causalnex.discretiser import Discretiser
```

## pgmpy import
```python
# ベイジアンネットワークが使える
import pgmpy
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore
```

# データ説明&読み込み
UCIのdatasetsから拝借(製造業関連のデータ)。  
[UCIサイトページ](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) のデータ説明翻訳。  
データセットは10,000点のデータが行として格納され、14の特徴が列として格納されている。
- UID：1～10000までの一意な識別子
- プロダクトID：低品質（全製品の50％）、中品質（30％）、高品質（20％）を表すL、M、Hの文字とバリエーション固有のシリアルナンバーから構成されている。
- 気温[K]：ランダムウォーク過程を用いて生成し、300K付近で標準偏差2Kに正規化したもの。
- プロセス温度[K]：標準偏差1Kに正規化したランダムウォーク過程を用いて生成し、気温に10Kを加算した。
- 回転数 [rpm]: 2860Wの電力から計算され、正規分布のノイズが重畳されている。
- トルク[Nm]：トルク値は40Nmを中心に正規分布し、φ＝10Nm、負の値はない。
- 工具摩耗量 [分]： H/M/Lは、プロセスで使用される工具に5/3/2分の工具摩耗を追加する。
- 'machine failure' ラベルは、この特定のデータポイントにおいて、以下の故障モードのいずれかが真である場合に、機械が故障したかどうかを示す。

マシン故障は、以下の5つの独立した故障モードで構成される。
- 工具摩耗故障（TWF）：工具摩耗時間が200～240分の間でランダムに選択された時点で、工具が故障して交換される（データセットでは120回）。この時点で、工具は69回交換され、51回故障する（ランダムに割り当てられる）。
- 放熱不良（HDF）：空気温度と加工温度の差が8.6K以下で、工具の回転速度が1380rpm以下の場合、放熱が原因で加工不良となる。115データポイントにおいて、このケースに該当する。
- power failure (PWF)：トルクと回転速度（rad/s）の積が、プロセスに必要な電力に等しい。この電力が3500W以下または9000W以上の場合、プロセスは失敗し、これはデータセットで95回のケースである。
- 過ひずみ故障（OSF）：工具摩耗とトルクの積が、L製品バリエーション（12000M、13000H）において11000minNmを超えた場合、過ひずみによりプロセスが失敗する。これは98個のデータポイントに当てはまる。
- ランダム故障（RNF）：各プロセスは、プロセスパラメータに関係なく、0.1 %の確率で故障する。この確率は5データポイントのみであり、データセットの10,000データポイントに対して予想される確率よりも低い。

上記の故障モードのうち少なくとも1つが真であれば、プロセスは失敗し、「機械故障」ラベルは1に設定される。

```python
# データ読み込み
df = pd.read_csv('dataset/ai4i2020.csv')#, parse_dates=['Time'])
dfDummy = pd.get_dummies(df[['Type']])
df = pd.concat([df, dfDummy], axis=1)
#df.drop(columns=['Type'], inplace=True)
print(df.shape)
print('ALL NaN Count', df.isnull().sum().sum())
print(df.columns)
display(df.head(10))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7e83fb23-cc51-c614-23fb-6068505bc29f.png)

# 前処理
```python
# カラムリスト定義
colsname = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]'
            , 'Torque [Nm]', 'Tool wear [min]'
            , 'Type_H','Type_L', 'Type_M'
            , 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
numcols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]'
           , 'Torque [Nm]', 'Tool wear [min]']
catcols = ['Type_H','Type_L', 'Type_M'
           , 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
```

非線形Lingamなど10000サンプルだと計算が終わらないアルゴリズムもあったので、サンプルサイズを1000まで減らす。
```python
# データ量が多いのでサンプリング
# sampling
df_sample_, df_sample = sklearn.model_selection.train_test_split(df
                                                                 , test_size=0.1
                                                                 , stratify=df['Machine failure'], shuffle=True, random_state=0)
display(df.describe())
display(df_sample_.describe())
display(df_sample.describe())
```
全データ10000サンプル基本統計量
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ff138fc6-0b94-5d5e-48c1-ca1a7f27b8a3.png)
9000サンプル基本統計量
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/62013806-9b16-1b07-a442-957b5a48482d.png)
1000サンプル基本統計量
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7375ca67-7385-379b-e93f-61e4d805bf9a.png)

連続値以外に2値変数もあるが、特に構わずすべて標準化する。
```python
# df_sampleを標準化
df_std = df_sample[colsname].copy()
ss = sklearn.preprocessing.StandardScaler()
df_std = pd.DataFrame(ss.fit_transform(df_std), columns=colsname)
display(df_std)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/eaee14b9-0b07-5d91-0bca-b9dc2e4d13fa.png)

# 想定される因果(実際は不明)
UCIのデータの説明から筆者が考えた因果関係を示す。
```python
# 想定される因果
G = nx.DiGraph()  # 有向グラフ (Directed Graph)
# 頂点の追加
#G.add_node('Air temperature [K]')                
G.add_nodes_from(colsname)

# 辺の追加 (頂点も必要に応じて追加される)
#G.add_edge(1, 2)                                    
G.add_edges_from([('Air temperature [K]', 'HDF')
                  , ('Air temperature [K]', 'Process temperature [K]')
                  , ('Process temperature [K]', 'HDF')
                  , ('Rotational speed [rpm]', 'Process temperature [K]')
                  , ('Rotational speed [rpm]', 'Tool wear [min]')
                  , ('Rotational speed [rpm]', 'PWF')
                  , ('Rotational speed [rpm]', 'Torque [Nm]')
                  , ('Torque [Nm]', 'PWF')
                  , ('Torque [Nm]', 'OSF')
                  , ('Tool wear [min]', 'OSF')
                  , ('Tool wear [min]', 'TWF')
                  , ('Type_H', 'Tool wear [min]')
                  , ('Type_L', 'Tool wear [min]')
                  , ('Type_M', 'Tool wear [min]')
                 ])

plt.figure(figsize=(15,10))
pos = nx.circular_layout(G, scale=1, center=None, dim=2)  # ここのポジションをこの先のplotでも使用する
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()
#svg = SVG(nx.nx_agraph.to_agraph(G).draw(prog='fdp', format='svg'))
#display(svg)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/0de01866-3465-55c3-c19c-96d3fe9baa0b.png)

# 軽くデータ俯瞰
## 全データでの俯瞰
相関行列
```python
# 相関行列
corr_ = df[numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].corr().to_numpy()-np.diag(np.diag(df[numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].corr()))
corr_ = pd.DataFrame(corr_, index=numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], columns=numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
display(corr_.style.bar(color='lightcoral'))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/1fceeae7-b966-cab4-ff10-470800674efa.png)

相関比
```python
# 相関比
def CorrelationV(data:'dataframe', cat_name:'categorical', num_name:'numerical'):
    """
    Calc Correlation ratio 

    Parameters
    ----------
    data : DataFrame
    x : categorical
    y : numerical
    """
    datadrop = data.dropna()
    x = datadrop[cat_name].to_numpy()
    y = datadrop[num_name].to_numpy()
    variation = ((y - y.mean()) ** 2).sum()
    inter_class = sum([((y[x == i] - y[x == i].mean()) ** 2).sum() for i in np.unique(x)])
    correlation_ratio = inter_class / variation
    return 1 - correlation_ratio

results = []
for n_c in numcols:
    for c_c in catcols:
        results.append([n_c, c_c, CorrelationV(df, c_c, n_c)])
results = pd.DataFrame(results, columns=['numerical', 'categorical', 'CorrelationRatio']).sort_values('CorrelationRatio', ascending=False)
display(results)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6ce9781e-a748-2e35-4b4d-3ab2225f53e2.png)

pairplot
```python
# pairplot
sns.pairplot(df[numcols+['Type_H', 'Type_M','Type_L', 'Machine failure']], hue='Machine failure')
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/caaf718e-2ca3-d058-eaeb-b5b239c061d7.png)

## サンプルサイズを1000まで減らした後の俯瞰
相関行列
```python
# 相関行列
corr_ = df_sample[numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].corr().to_numpy()-np.diag(np.diag(df_sample[numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].corr()))
corr_ = pd.DataFrame(corr_, index=numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], columns=numcols+['Type_H', 'Type_M','Type_L', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
display(corr_.style.bar(color='lightcoral'))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/a350450e-e45b-ed65-810e-2ec1c8facd0f.png)

相関比
```python
# 相関比
results = []
for n_c in numcols:
    for c_c in catcols:
        results.append([n_c, c_c, CorrelationV(df_sample, c_c, n_c)])
results = pd.DataFrame(results, columns=['numerical', 'categorical', 'CorrelationRatio']).sort_values('CorrelationRatio', ascending=False)
display(results)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ff9329c8-bb12-62d1-84d5-420ffbf3b09c.png)


pairplot
```python
# pairplot
sns.pairplot(df_sample[numcols+['Type_H', 'Type_M','Type_L', 'Machine failure']], hue='Machine failure')
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7f2e8027-44c5-decd-1b83-ac643d24890d.png)

# LiNGAM関連
推定アルゴリズムとしては，まず独立成分分析によるアプローチ [Shimizu06] が提案された。
その後、回帰分析と誤差項の独立性評価を繰り返し、誤差項同士の相関の大きさを最小化するように因果的順序と係数行列を決定する“DirectLiNGAM”という手法が提案された [Shimizu11]。また、

- 時系列データで時間差を伴う効果がある場合の，ベクトル自己回帰モデルを用いた分析手法[Hyvarinen10]
- ブートストラップ法による再標本化と DirectLiNGAM の実行を繰り返すことによる，因果関係の有無・係数値の統計的信頼性の評価法 [Komatsu10, Thamvitayakul12]
- 未観測共通要因がある場合でも因果グラフの全体像を推定する手法 [Maeda20]等にも派生する

など、LiNGAM はその仮定を緩めつつ適用可能範囲を拡大している。
これらの派生形も含め、PythonのLiNGAMパッケージはGitHub上で公開されており、データセットを正しく構築しさえすれば、tutorialに沿ってjupyter lab等で所定のコードを入力して実行するだけで、結果が出力される。（https://github.com/cdt15/lingam）

[博士課程進学率に関する因果モデルの構築 -統計的因果探索アルゴリズム"LiNGAM"による試行的分析-](https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/1) より抜粋

## DirectLiNGAM
https://lingam.readthedocs.io/en/latest/tutorial/lingam.html

DirectLiNGAMは、基本的なLiNGAMモデルを直接学習する方法である。
誤差変数間の独立性を評価するために、エントロピーに基づく尺度を用いる。
基本的なLiNGAMモデルは、以下のような仮定を置いている。
- 直線性
- 非ガウス型連続誤差変数（最大1つを除く）
- 非周期性
- 隠れた共通原因がない

観測変数を$x_i$、誤差変数を$e_i$、係数または接続強度を$b_{ij}$とする。
これらをそれぞれベクトル$x$、$e$、行列$B$にまとめる。
非周期性の仮定により、隣接行列$B$は行と列の同時並べ替えによって厳密に下三角形になるように並べ替えることができる。
誤差変数$e_i$は、隠れた共通原因がないという仮定により、独立である。
そして、数学的には観測変数ベクトル$x$のモデルは次のように書かれる。

$x=Bx+e$

(以下 [「Just-In-Timeモデルを利用した非定常非線形時系列データに対する因果探索」](https://www.jstage.jst.go.jp/article/pjsai/JSAI2022/0/JSAI2022_3E4GS205/_pdf/-char/ja) から引用)  
この$B$を重みつき有向グラフの隣接行列と見た時、その有向グラフが非巡回であることも仮定している。  
$b_{ij}\neq{0}$であることは$x_j→x_i$の向きに因果関係があることを意味する。  
LiNGAMにおける因果探索とは、この$B$を求めることと等価である。  
※「$b_{ij}\neq{0}$であることは$x_j→x_i$の向きに因果関係がある」について、以下の式のように$j$から$i$への因果となる。
```math
x_i=\sum_{j=1,j\neq{i}}^{P} b_{ij}x_j + e_i　　(i=1,2,\cdots, P)
```
学習
```python
%%time
# DirectLiNGAM
print(time.ctime())
model = lingam.DirectLiNGAM()
model.fit(df_std)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_DLingam.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(15,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = dict([((u,v,), f"{d['weight']:.3f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/df4d9f74-6b4d-52c0-ca99-661529f94be3.png)

`make_dot()`でも可視化できる。
```python
# 隣接行列を有向グラフで可視化(make_dot使用)
print(model.causal_order_)
print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)
make_dot(model.adjacency_matrix_, labels=colsname)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/b1ef9f01-82f4-1732-f87b-5f41ecd1caa9.png)

## DirectLiNGAM bootstrap
https://lingam.readthedocs.io/en/latest/tutorial/bootstrap.html

学習
```python
%%time
# DirectLiNGAM bootstrap
print(time.ctime())
n_sampling = 100
model = lingam.DirectLiNGAM()
result = model.bootstrap(df_std, n_sampling=n_sampling)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_DLingam_bootstrap.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/3dfdd039-afbe-6aee-fdff-a2d7435c9614.png)

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
print(model.causal_order_)
print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)

adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(15,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7823ed51-4e65-9664-44c2-dabed42a9eb9.png)

```python
# 隣接行列を有向グラフで可視化(make_dot使用)
make_dot(model.adjacency_matrix_, labels=colsname)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/fc86fc5e-447b-7060-b8c4-93dbc9bff994.png)

`get_causal_direction_counts()`によって抽出された因果方向のランキングを取得できる。
以下のコードでは、n_directions上位10位までの因果方向に選択肢を限定し、min_causal_effect係数0.1以上の因果方向に選択肢を限定している。
```python
%%time
# DirectLiNGAM bootstrap result
print(time.ctime())
cdc = result.get_causal_direction_counts(n_directions=10, min_causal_effect=0.1, split_by_causal_effect_sign=True)
print_causal_directions(cdc, n_sampling)
print(time.ctime())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/3fdd242d-d77c-7ce4-d308-1dd0741caba4.png)

`get_directed_acyclic_graph_counts()`メソッドを使用して、抽出された DAG のランキングを取得できる。
以下のコードでは、n_dagsオプションはランキング上位5のダグに限定され、min_causal_effectオプションは係数0.1以上の因果方向に限定される。
```python
%%time
# DirectLiNGAM bootstrap result
print(time.ctime())
dagc = result.get_directed_acyclic_graph_counts(n_dags=5, min_causal_effect=0.1, split_by_causal_effect_sign=True)
print_dagc(dagc, n_sampling)
print(time.ctime())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6e661f3b-e60e-230f-004e-d4486847d34b.png)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e48c8f39-03d8-4545-f515-60a10f265e36.png)

`get_probabilities()`を使用して、ブートストラップの確率を取得できる。
```python
%%time
# DirectLiNGAM bootstrap result
print(time.ctime())
prob = result.get_probabilities(min_causal_effect=0.1)
prob = np.where(prob<0.01, 0, prob)
prob = pd.DataFrame(prob, columns=colsname, index=colsname)
display(prob)
print(time.ctime())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/47bb69a2-0051-e3eb-c74c-0f14a8771547.png)

```python
# ブートストラップ確率可視化
G=nx.from_pandas_adjacency(prob.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/71eec510-3b98-87f6-38b5-b0c5b6628ad1.png)

## RESIT
https://lingam.readthedocs.io/en/latest/tutorial/resit.html

RESITは、Additive Noise Modelの推定アルゴリズムである。
この方法は、以下のような仮定をおいている。
- 連続変数
- 非線形性
- 加法性ノイズ
- 非周期性
- 隠れた共通原因(未観測共通因子)がない

観測変数を$x_i$、誤差変数を$e_i$と表記する。
誤差変数$e_i$は、隠れた共通原因がないという仮定により独立である。すると、数学的には観測変数$x_i$のモデルは次のように表記される。
```math
x_i=f_i(pa(x_i))+e_i
```
ここではある非線形（微分可能）関数であり、$pa(x_i)$は$x_i$の親である。

学習
```python
%%time
# RESIT
print(time.ctime())
reg = RandomForestRegressor(max_depth=4, random_state=0)

model = lingam.RESIT(regressor=reg)
model.fit(df_std)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_RESIT.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e270618b-ed9e-f50c-e6f7-50057e64c5ca.png)

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
print(model.causal_order_)
print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)

adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.1f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()

# 隣接行列を有向グラフで可視化(make_dot使用)
#make_dot(model.adjacency_matrix_, labels=colsname)
```
非線形のアプローチなので、エッジが有る/無いの1, 0での表現。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/462c50d4-e6d1-44d1-d39d-6f02da7f9f48.png)

## RESIT bootstrap
https://lingam.readthedocs.io/en/latest/tutorial/bootstrap.html

学習
```python
%%time
# RESIT bootstrap
print(time.ctime())
n_sampling = 100
model = lingam.RESIT(regressor=reg)
result = model.bootstrap(df_std, n_sampling=n_sampling)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_RESIT_bootstrap.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/89462032-fc4c-2c55-30dd-cae90bd42c1b.png)

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
print(model.causal_order_)
print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)

adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.1f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()

# 隣接行列を有向グラフで可視化(make_dot使用)
#make_dot(model.adjacency_matrix_, labels=colsname)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/187c0a23-d549-752c-14fc-4f57b54d4318.png)

```python
%%time
# RESIT bootstrap result
print(time.ctime())
cdc = result.get_causal_direction_counts(n_directions=10, min_causal_effect=0.01, split_by_causal_effect_sign=True)
print_causal_directions(cdc, n_sampling)
print(time.ctime())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d66b68ba-c887-f22e-fbb9-178e6dba8bc4.png)


```python
%%time
# RESIT bootstrap result
print(time.ctime())
dagc = result.get_directed_acyclic_graph_counts(n_dags=5, min_causal_effect=0.1, split_by_causal_effect_sign=True)
print_dagc(dagc, n_sampling)
print(time.ctime())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e4be4b0b-675f-9c13-1994-d1f775bf09ac.png)

```python
%%time
# RESIT bootstrap result
print(time.ctime())
prob = result.get_probabilities(min_causal_effect=0.1)
prob = np.where(prob<0.5, 0, prob)
prob = pd.DataFrame(prob, columns=colsname, index=colsname)
display(prob)
from_index = 0 # index of x0
to_index = 12 # index of x3
# 開始変数から終了変数までのすべてのパスとそのブートストラップ確率を取得
display(pd.DataFrame(result.get_paths(from_index, to_index)))
print(time.ctime())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/0e1b9b1f-7d1d-58f3-c17f-e79151f15309.png)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6e7be73b-ee35-f92a-82ee-d9ca51b9bdd0.png)

```python
# ブートストラップ確率可視化
G=nx.from_pandas_adjacency(prob.T, create_using=nx.DiGraph)
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/2f619311-b0bd-ea87-610a-f55febebecb5.png)

## RCD
https://lingam.readthedocs.io/en/latest/tutorial/rcd.html

本手法RCD（Repetitive Causal Discovery）は、基本LiNGAMモデルを隠れ共通原因ケースに拡張したもの、すなわち潜在変数LiNGAMモデルを前提としている。
基本的なLiNGAMモデルと同様に、この方法は以下の仮定をする：
- 直線性
- 非ガウス連続誤差変数
- 非周期性

しかし、RCDは隠れた共通原因の存在を許容する。
この場合、双方向の弧は同じ隠れた共通原因を持つ変数の組を示し、有向矢印は同じ隠れた共通原因の影響を受けない変数の組の因果方向を示す因果グラフが出力される。

学習
```python
%%time
# RCD
print(time.ctime())
model = lingam.RCD()
model.fit(df_std)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_RCD.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/526f51f1-8602-ecfc-c671-c8d94c9ac232.png)

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
#print(model.causal_order_)
#print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)

adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.1f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()

# 隣接行列を有向グラフで可視化(make_dot使用)
#make_dot(model.adjacency_matrix_, labels=colsname)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/66ed0eb1-235a-602b-821a-b9fb56e56ba4.png)

## RCD bootstrap
https://lingam.readthedocs.io/en/latest/tutorial/bootstrap.html

学習
```python
%%time
# RCD bootstrap
print(time.ctime())
n_sampling = 100
model = lingam.RCD()
result = model.bootstrap(df_std, n_sampling=n_sampling)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_RCD_bootstrap.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9195d371-bf15-d7d0-e1df-ec453b76c69e.png)

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
#print(model.causal_order_)
#print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)

adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.1f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()

# 隣接行列を有向グラフで可視化(make_dot使用)
#make_dot(model.adjacency_matrix_, labels=colsname)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/82525113-a74b-9f91-c1d4-503af46b90e1.png)

## CAM-UV
https://lingam.readthedocs.io/en/latest/tutorial/camuv.html

本手法CAM-UV（Causal Additive Models with Unobserved Variables）は、基本的なCAMモデルを拡張して、未観測変数を含めることを想定している。
この方法は、次のような仮定をする：
- 変数形式の一般化加法モデル (GAM) に対する直接原因の影響。
- 因果構造は有向非巡回グラフ (DAG) を形成する。

CAM-UVは、未観測変数の存在を許容する。
無向の辺は、観察されない因果パス（UCP）または観察されないバックドアパス（UBP）を持つ変数の組を示し、有向の辺は、UCPまたはUBPを持たない変数の組の因果方向を示す因果グラフを出力する。
UCPとUBPの定義：
下図に示すように、$x_j$から$x_i$への因果経路は、$x_i$とその未観測の直接原因を結ぶ有向辺で終わっている場合、UCPと呼ばれる。
$x_i$と$x_j$の間の裏道は、$x_i$とその観測されていない直接原因を結ぶ辺から始まり、$x_j$とその観察されない直接原因を結ぶ辺で終わるなら、UBPと呼ばれる。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9df8eadb-fe6a-3b43-44aa-d1d51dbd13f6.png)
(図：[CAM-UV](https://lingam.readthedocs.io/en/latest/tutorial/camuv.html)より)

※bootstrapはない

学習
```python
%%time
# CAM-UV
print(time.ctime())
model = lingam.CAMUV()
model.fit(df_std)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_CAMUV.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e93f2a0f-6a88-a8d6-5909-553470443e6d.png)

隣接行列を有向グラフで可視化
```python
# 隣接行列を有向グラフで可視化(networkx使用)
#print(model.causal_order_)
#print(np.array(colsname)[model.causal_order_])
#print(model.adjacency_matrix_)

adjacency_ = pd.DataFrame(model.adjacency_matrix_, columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_.T, create_using=nx.DiGraph)  # 列が原因、行が結果なのでnetworkxでは転置の必要がある
plt.figure(figsize=(16,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.1f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()

# 隣接行列を有向グラフで可視化(make_dot使用)
#make_dot(model.adjacency_matrix_, labels=colsname)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/51c1741f-15be-ba2c-3563-394482209c5b.png)

## LiM
__※1000サンプルだと計算が終わらず、34サンプルまでデータ量を減らしてもMemoryErrorが出る__
https://lingam.readthedocs.io/en/latest/tutorial/lim.html

線形混合(LiM)因果発見アルゴリズムは、連続変数と離散変数の両方からなる混合データを扱えるようにLiNGAMを拡張したものである。
推定は、まず非周期性制約のあるデータの結合分布に対する対数尤度関数を大域的に最適化し、次に局所的な組み合わせ探索を適用して因果関係グラフを出力することで行われる。
本手法は、以下に示すLiMモデルに基づいている、

i) 連続変数については、$x_i$のそれぞれに割り当てられるその値は、$x_{pa(i)}$で示されるその親変数に非ガウス誤差項$e_i$を加えた線形関数、即ち
$$
x_i=e_i+c_i+\sum_{j\in{pa(i)}}b_{ij}x_j,　e_i ～ Non-Gaussian(⋅)
$$ここで、誤差項$e_i$は非ガウス密度を持つ連続的な確率変数であり、誤差変数$e_i$は互いに独立である。係数$b_{ij}$と切片$c_i$は定数である。

ii) 離散変数については、その親変数$x_{pa(i)}$にロジスティック誤差項$e_i$を加えた線形関数が0より大きければその値は1に等しく、そうでなければその値は0に等しい、すなわち、

$$ x_i=
    \begin{cases}
        {1, \  e_i+c_i+\sum_{j\in{pa(i)}}b_{ij}x_j>0}\\
        {0, \  otherwise}
    \end{cases}
    ,　e_i ～ Logistic(0,1)
$$ここで、誤差項$e_i$はロジスティック分布に従うが、その他の表記は連続変数における表記と同じである。
本手法では、以下の前提を設けている。
- 連続変数とバイナリ変数
- 直線性
- 非周期性
- 隠された一般的な原因(未観測共通要因)がない
- バイナリ変数のすべてのペアについて、一方のバイナリ変数を他方から予測する場合、ベースラインは同じ


少し前処理
```python
# 連続値変数だけ標準化
df_nonstd = df_sample[colsname].reset_index(drop=True).copy()
#ss = sklearn.preprocessing.StandardScaler()
#df_nonstd_num = pd.DataFrame(ss.fit_transform(df_nonstd[numcols]), columns=numcols)
#df_nonstd_cat = df_nonstd[catcols]
#df_nonstd = pd.concat([df_nonstd_num, df_nonstd_cat], axis=1)
df_nonstd[numcols] = df_nonstd[numcols].astype(float)
for i in colsname:
    print(i,':', sklearn.utils.multiclass.type_of_target(df_nonstd[i]))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/2e3206c6-e470-4045-7382-60179e463651.png)
```python
# 連続値か離散値かのフラグ
# 1:continuous;   0:discrete
dis_con = np.array([[0 if len(df_std[c].unique())<5 else 1 for c in df_std.columns]])
dis_con
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/2a0277b6-099b-0e17-25db-af6f0ba9e514.png)

学習
```python
%%time
# LiM
# どれだけデータ量を減らしてもMemoryErrorが出る
#print(time.ctime())
#model = lingam.LiM()
#model.fit(df_nonstd[(df_nonstd['TWF']==1)|(df_nonstd['HDF']==1)|(df_nonstd['PWF']==1)|(df_nonstd['OSF']==1)|(df_nonstd['RNF']==1)].to_numpy(),  dis_con)
#print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_LiM.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/259e8c7e-bc09-550c-136e-fc9c707c74b5.png)

# NOTEARS
https://causalnex.readthedocs.io/en/latest/

https://github.com/quantumblacklabs/causalnex/

[日本語記事:PythonでNOTEARS・ベイジアンネットによる因果グラフ推定 -causalnexの紹介-](https://yuminaga.hatenablog.com/entry/2021/05/23/135852)
[日本語記事:Pythonによる因果グラフ推定 -causalnexの紹介 その2-](https://yuminaga.hatenablog.com/entry/2021/05/23/170842)
[日本語記事:因果探索ライブラリcausalnex](https://socinuit.hatenablog.com/entry/2021/03/27/224414)

「DAGs with NO TEARS:Continuous Optimization for Structure Learning」
Abstract：
DAGの構造を推定することは、DAGの探索空間が組合せ的であり、ノード数に応じて超指数関数的にスケールするため、困難な問題である。
既存のアプローチは、非周期性制約を強制するための様々な局所的ヒューリスティックに依存している。
この組み合わせ制約を完全に回避するために、構造学習問題を実数行列に対する純粋な連続最適化問題として定式化することで、根本的に異なる戦略を導入する。
これは、非周期性の新しい特徴づけによって達成され、滑らかであるばかりでなく、厳密でもある。
その結果、この問題は標準的な数値アルゴリズムで効率的に解くことができ、実装も容易となる。
提案手法は、グラフに構造的な仮定（樹幅や次数の制限など）を課すことなく、既存の手法を凌駕する。
[論文「DAGs with NO TEARS:Continuous Optimization for Structure Learning」](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)  
[Qiita記事「DAG の構造学習を連続最適化問題に落とし込んで解くNO TEARSアルゴリズム」](https://qiita.com/kueda_cs/items/5b163bd778abe1b109e8)

## NOTEARSによる基本的な構造学習
学習
```python
%%time
# NOTEARSによる基本的な構造学習
print(time.ctime())
sm = from_pandas(df_std, tabu_edges = [], tabu_parent_nodes = None, tabu_child_nodes = None,)
# DAGになるように閾値をあげる
sm.threshold_till_dag()
# 係数の閾値を設定
sm.remove_edges_below_threshold(0.1)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_NOTEARS.pickle', mode='wb') as f:
#    pickle.dump(sm, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/4e89c38e-1d81-9dcd-2e05-c7c5510ff722.png)

有向グラフで可視化
```python
# CausalNexのグラフ描画機能
#from causalnex.plots import plot_structure
#from IPython.display import Image
#viz = plot_structure(sm)  # Default CausalNex visualisation
#image_binary = viz.draw(format="jpg", prog="twopi")
#Image(image_binary,width=1000)

# 可視化
plt.figure(figsize=(15,10))
#pos = nx.circular_layout(sm, scale=1)
#pos = nx.nx_agraph.graphviz_layout(sm, prog="dot")
nx.draw_networkx_edge_labels(sm, pos
                             , edge_labels={(u, v): round(d["weight"], 4) for (u,v,d) in sm.edges(data=True)}
                             #, font_color=edge_weights_color
                             #, font_size = edge_weights_fontsize
                            )
# edgeの重みに応じて太さを変更、最低でもmin_edge_widthに設定
#edge_width = [np.min([np.max([np.abs(d["weight"]), 0.1]), 1]) for (u, v, d) in sm.edges(data=True)]
nx.draw_networkx(sm, pos = pos, with_labels = True
#                 , width = edge_width
                )
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9a2ad51b-bc2b-1090-1ada-16df263d3783.png)

## NOTEARSによるLassoを用いた構造学習
学習
```python
%%time
print(time.ctime())
# NOTEARSを実行, from_pandas_lassoでL1 penaltyをつけて推定することが可能
sm = from_pandas_lasso(df_std
                       , beta = 0.01 # L1 penalty の強さ
                       , tabu_edges = [], tabu_parent_nodes = None, tabu_child_nodes = None,)
# DAGになるように閾値をあげる
sm.threshold_till_dag()
# 係数の閾値を設定
sm.remove_edges_below_threshold(0.1)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_NOTEARS_lasso.pickle', mode='wb') as f:
#    pickle.dump(sm, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d2ed445f-20a2-0ec5-d95d-cf9b83ebf973.png)

有向グラフで可視化
```python
# 可視化
plt.figure(figsize=(15,10))
#pos = nx.spring_layout(sm, seed=0)
#pos = nx.nx_agraph.graphviz_layout(sm, prog="dot")
nx.draw_networkx_edge_labels(sm, pos
                             , edge_labels={(u, v): round(d["weight"], 4) for (u,v,d) in sm.edges(data=True)}
                             #, font_color=edge_weights_color
                             #, font_size = edge_weights_fontsize
                            )
# edgeの重みに応じて太さを変更、最低でもmin_edge_widthに設定
#edge_width = [np.min([np.max([np.abs(d["weight"]), 0.1]), 1]) for (u, v, d) in sm.edges(data=True)]
nx.draw_networkx(sm, pos = pos, with_labels = True
#                 , width = edge_width
                )
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/a61e1e17-8b9e-3236-cc8a-72e56672ba6e.png)

## NOTEARSによるPytorchを用いた構造学習
### Pytorchを用いた構造学習では離散値や連続値を指定できる
```python
# 連続値変数だけ標準化
df_nonstd = df_sample[colsname].reset_index(drop=True).copy()
ss = sklearn.preprocessing.StandardScaler()
df_nonstd_num = pd.DataFrame(ss.fit_transform(df_nonstd[numcols]), columns=numcols)
df_nonstd_cat = df_nonstd[catcols]
df_nonstd = pd.concat([df_nonstd_num, df_nonstd_cat], axis=1)
df_nonstd[numcols] = df_nonstd[numcols].astype(float)
for i in colsname:
    print(i,':', sklearn.utils.multiclass.type_of_target(df_nonstd[i]))
display(df_nonstd)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ffbe304e-3f1d-8196-7764-391de0af6739.png)

学習
```python
%%time
# NOTEARSを実行, from_pandas_pytorchでL1 penaltyやL2 penaltyをつけて推定することが可能
print(time.ctime())
# 連続値か離散値かカラムごとに指定
dist_type_schema = {c:'bin' if len(df_nonstd[c].unique())<5 else 'cont' for c in df_nonstd.columns}
# dist_type_schemaを加味してpytorchを用いて推定
sm = from_pandas_pytorch(df_nonstd
                         , dist_type_schema = dist_type_schema
                         , hidden_layer_units = None
                         , lasso_beta = 0.1
                         #, ridge_beta  = 0
                         #, w_threshold = 0.1
                         #, use_bias = False
                        )
# DAGになるように閾値をあげる
sm.threshold_till_dag()
# 係数の閾値を設定
sm.remove_edges_below_threshold(0.1)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_NOTEARS_pytorch.pickle', mode='wb') as f:
#    pickle.dump(sm, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f20b07b8-ddb4-c5ed-bd53-9c06b661a21d.png)

有向グラフで可視化
```python
# 可視化
plt.figure(figsize=(15,10))
#pos = nx.spring_layout(sm, seed=0)
#pos = nx.nx_agraph.graphviz_layout(sm, prog="dot")
nx.draw_networkx_edge_labels(sm, pos
                             , edge_labels={(u, v): round(d["weight"], 4) for (u,v,d) in sm.edges(data=True)}
                             #, font_color=edge_weights_color
                             #, font_size = edge_weights_fontsize
                            )
# edgeの重みに応じて太さを変更、最低でもmin_edge_widthに設定
#edge_width = [np.min([np.max([np.abs(d["weight"]), 0.1]), 1]) for (u, v, d) in sm.edges(data=True)]
nx.draw_networkx(sm, pos = pos, with_labels = True
#                 , width = edge_width
                )
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/83d631f5-7835-a387-c4cf-460b2aba2d88.png)

### 連続値や離散値の指定なく実行
学習
```python
%%time
# NOTEARSを実行, from_pandas_pytorchでL1 penaltyやL2 penaltyをつけて推定することが可能
print(time.ctime())
# 連続値か離散値かカラムごとに指定
dist_type_schema = {c:'bin' if len(df_nonstd[c].unique())<5 else 'cont' for c in df_nonstd.columns}
# dist_type_schemaを加味してpytorchを用いて推定
sm = from_pandas_pytorch(df_std
                         #, dist_type_schema = dist_type_schema
                         , hidden_layer_units = None
                         , lasso_beta = 0.01
                         #, ridge_beta  = 0
                         #, w_threshold = 0.1
                         #, use_bias = False
                        )
# DAGになるように閾値をあげる
sm.threshold_till_dag()
# 係数の閾値を設定
sm.remove_edges_below_threshold(0.1)
print(time.ctime())

# DAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_NOTEARS_pytorch2.pickle', mode='wb') as f:
#    pickle.dump(sm, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/898be73d-fe18-ada1-0c6f-7322d1216c61.png)

有向グラフで可視化
```python
# 可視化
plt.figure(figsize=(15,10))
#pos = nx.spring_layout(sm, seed=0)
#pos = nx.nx_agraph.graphviz_layout(sm, prog="dot")
nx.draw_networkx_edge_labels(sm, pos
                             , edge_labels={(u, v): round(d["weight"], 4) for (u,v,d) in sm.edges(data=True)}
                             #, font_color=edge_weights_color
                             #, font_size = edge_weights_fontsize
                            )
# edgeの重みに応じて太さを変更、最低でもmin_edge_widthに設定
#edge_width = [np.min([np.max([np.abs(d["weight"]), 0.1]), 1]) for (u, v, d) in sm.edges(data=True)]
nx.draw_networkx(sm, pos = pos, with_labels = True
#                 , width = edge_width
                )
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/0bfd895d-7487-6c81-8aca-682eb453070f.png)

# Bayesian network
https://pgmpy.org/index.html  

[What are Bayesian Models](https://pgmpy.org/detailed_notebooks/2.%20Bayesian%20Networks.html#)  
他、小川雄太郎著 [「Pythonによる因果分析」](https://www.amazon.co.jp/dp/4839973571) が参考になる

確率的なグラフモデル（統計モデルの一種）で、ランダム変数の集合とその条件依存関係を有向非循環グラフ（DAG）を介して表現する。
ベイジアンネットワークは、確率変数間の因果関係を表現したい場合に多く使用される。
ベイジアンネットワークは、条件付き確率分布（CPD）を用いてパラメータ化される。
ネットワークの各ノードは$P(node|Pa(node))$を用いてパラメータ化され、$Pa(node)$はネットワーク内のノードの親を表す。  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f7336f4b-d75f-5098-2956-102792e8b37f.png)
(図：[What are Bayesian Models](https://pgmpy.org/detailed_notebooks/2.%20Bayesian%20Networks.html#)より)

__～[Learning Bayesian Networks from Data](https://pgmpy.org/detailed_notebooks/10.%20Learning%20Bayesian%20Networks%20from%20Data.html)より抜粋 & 筆者が一部追記～__  
ベイジアンネットワークはDAGが既知でパラメータ（個々の変数の(条件付き)確率分布）を学習することが多いが、構造学習を行うとデータドリブンにDAGを推定できる。
よってベイジアンネットワークの学習は、2つの問題に分けることができる：
- パラメータ学習：データサンプルの集合と変数間の依存関係を捉えたDAGが与えられたとき、個々の変数の（条件付き）確率分布を推定する
- 構造学習：データサンプルが与えられたら、変数間の依存関係を表すDAGを推定する

因果探索で重要な構造学習について、データセットからモデルの構造（DAG）を学習するためには、大きく分けて3つの手法がある。
- スコアベースの構造学習
- 制約に基づく構造学習
- 両技術を組み合わせる(ハイブリッド構造学習)

__スコアベース__  
モデルとデータの適合度を測るスコアとしてよく使われるのは、BDeuやK2などのベイズディリクレスコアとベイズ情報量基準（BIC、MDLとも呼ばれる）。  
DAGの探索空間は変数数に対して非常に指数関数的であり、スコアリング関数は局所最大を許容する。それによる性質は2つある。
- 最初の性質は、非常に小さいネットワークを除くすべてのネットワークで網羅的な検索を困難にする。  
- 2番目の性質は、局所最適化アルゴリズムが常に最適な構造を見つけることができないため、効率的な局所最適化ができない  

よって、理想的な構造を特定することは困難であるが、発見的探索戦略はしばしば良い結果をもたらす。

__制約に基づく構造学習__  
データからDAGを構築するための別の、しかし非常に簡単なアプローチは以下。
- 仮説検定を使ってデータセットの独立性を特定する
- 特定された独立性に従ってDAG（パターン）を構築する

データにおける独立性は、chi2条件付き独立性検定を用いて確認することができる。
そのために、変数Zsのセットが与えられたときに、XがYから独立しているかどうかを確認する条件付き独立性仮説検定が実行される。
独立性の検証方法が手元にあれば、3つのステップでデータセットからDAGを構築することができる。
1. 無向性のスケルトンを構築する - estimate_skeleton()
2. 部分有向アシクリッドグラフ(PDAG; DAGのI等価クラス)を得るために、強制された辺を配向する - skeleton_to_pdag()
3. DAGパターンを、残りの辺を何らかの方法で保守的に方向付けることでDAGに拡張する - pdag_to_dag()

ステップ1.&2.は、いわゆる[PCアルゴリズム](https://speakerdeck.com/s1ok69oo/pcarugorizumuniyorubeiziannetutowaku)。
PDAGはDirectedGraphであり、エッジの向きが決定されていないことを示すために、双方向のエッジを含むことができる。

__ハイブリッド構造学習__  
今回採用するのはこれ。
MMHCアルゴリズムは、制約に基づく方法とスコアに基づく方法を組み合わせたものである。
これは2つの部分からなる：
- 制約に基づく構築手順MMPCを用いて無向グラフのスケルトンを学習する
- スコアベースの最適化（BDeuスコア＋modified hill-climb）を用いてエッジをオリエンテーションする

pgmpyのモジュールとしては、["MmhcEstimator()"](https://pgmpy.org/structure_estimator/mmhc.html)を使って、スケルトン構築 & エッジの方向づけを実施できる。  
["MmhcEstimator()"](https://pgmpy.org/structure_estimator/mmhc.html)を使ってスケルトンだけ構築して、["HillClimbSearch()"](https://pgmpy.org/structure_estimator/hill.html)でエッジの方向づけを行うというアプローチでもいい。

離散化など前処理
```python
# ベイジアンネットを推定するために離散化データを作成
discretised_data = df_sample[colsname].copy()
split_point = [round(i,1) for i in np.linspace(0.1, 0.9, 9)]
for cc in numcols:
    Disc = Discretiser(method="percentiles", percentile_split_points=split_point)
    discretised_data[cc] = Disc.fit_transform(discretised_data[cc].to_numpy())
colsname_label = [i+'_label' for i in colsname]
discretised_data.columns = colsname_label
display(discretised_data)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/0d906e69-5e64-6619-6bc7-6ce79f8a6eff.png)
```python
# 離散化した後各ラベルの平均値
check = pd.concat([df_sample[colsname], discretised_data], axis=1)
empty = pd.DataFrame()
for cc in numcols:
    empty = pd.concat([empty, check.groupby([cc+'_label'])[[cc]].mean()], axis=1)
print('mean')
display(empty)
empty = pd.DataFrame()
for cc in numcols:
    empty = pd.concat([empty, check.groupby([cc+'_label'])[[cc]].count()], axis=1)
print('count')
display(empty)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/fc761474-ba8d-6999-be50-eb4d6696cbdd.png)

MMHCスケルトンまでの構築なら以下（DAGは構築しない）
```python
%%time
# MMHCアルゴリズムによるスケルトン構築
print(time.ctime())
mmhc = MmhcEstimator(discretised_data)
skeleton = mmhc.mmpc()
# model_mmhc = mmhc.estimate()  # DAG構築までならこれを実施
print(time.ctime())

# スケルトンやDAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/skeleton.pickle', mode='wb') as f:
#    pickle.dump(skeleton, f)
# スケルトンやDAGをLoadするなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/skeleton.pickle', mode='rb') as f:
#    skeleton = pickle.load(f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/4baa49d2-7c95-e326-f1dc-55be548016f2.png)

MMHCアルゴリズムによるDAG構築なら以下（スケルトンで終わらずDAGまで構築）
```python
%%time
# MMHCアルゴリズムによるDAG構築
print(time.ctime())
mmhc = MmhcEstimator(discretised_data)
#skeleton = mmhc.mmpc()  # スケルトン構築だけならこれを実施
model_mmhc = mmhc.estimate()
print(time.ctime())

# スケルトンやDAGを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_mmhc.pickle', mode='wb') as f:
#    pickle.dump(model_mmhc, f)

# スケルトンやDAGをLoadするなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_mmhc.pickle', mode='rb') as f:
#    model_mmhc = pickle.load(f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/653828fb-9eba-7e89-bf33-87b80acebf85.png)

HillClimbSearchによる方向づけ
```python
%%time
# HillClimbSearchによる方向づけ
print(time.ctime())
hc = HillClimbSearch(discretised_data)
model = hc.estimate(tabu_length=15, white_list=skeleton.to_directed().edges(), scoring_method=BDeuScore(discretised_data))
print(time.ctime())
```

有向グラフで可視化(mmhcの結果)
```python
# mmhcのみの結果可視化
plt.figure(figsize=(15,10))
pos2 = {i+'_label':j for i, j in pos.items()}
#pos = nx.spring_layout(sm, seed=0)
#pos = nx.nx_agraph.graphviz_layout(sm, prog="dot")
nx.draw_networkx(model_mmhc, pos2, with_labels=True)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7a267fb6-360d-3732-8591-abab60643264.png)

有向グラフで可視化(mmhc＋HillClimbSearchの結果)
```python
# mmhc＋HillClimbSearchの結果可視化
plt.figure(figsize=(15,10))
pos2 = {i+'_label':j for i, j in pos.items()}
#pos = nx.spring_layout(sm, seed=0)
#pos = nx.nx_agraph.graphviz_layout(sm, prog="dot")
nx.draw_networkx(model, pos2, with_labels=True)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/952912e9-3248-df15-837d-17b0eff9a929.png)

# Graphical Lasso

https://scikit-learn.org/stable/modules/covariance.html

https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance

因果はわからないが、相関の有無がわかるGraphical Lassoによる無向グラフの構築も試してみる。

Graphical LassoはL1ペナルティ付き推定量を使用したスパース逆共分散推定。  
分散共分散行列の逆行列(精度行列)をスパース推定するアルゴリズム。  
精度行列から偏相関行列を求められ、それを構造とする。  
共分散行列の逆行列は、しばしば精度行列と呼ばれ、偏相関行列に比例する。  
これは部分独立の関係を与える。言い換えれば、2つの特徴が他の特徴に対して条件付きで独立している場合、精度行列の対応する係数はゼロになる。  
共分散行列の推定は、データから独立関係を学習することで、より良い条件付けができる。これは共分散選択と呼ばれる。  
n_samplesがn_featuresのオーダーかそれ以下の小さなサンプル数では、スパース逆共分散推定器は縮約共分散推定器よりもうまくいく傾向がある。  
しかし、その逆の状況や、非常に相関の強いデータでは、数値的に不安定になることがある。  
また、収縮推定量とは異なり、スパース推定量は対角線外の構造を回復することができる。  
GraphicalLasso推定器は，L1ペナルティを用いて精度行列のスパース性を強制する。  
そのαパラメータが高いほど，精度行列はよりスパースとなる。  
対応するGraphicalLassoCVオブジェクトはクロスバリデーションを用いてalphaパラメータを自動的に設定する。

※
基礎となるグラフに、平均的なノードよりもはるかに多くの接続を持つノードがある場合、アルゴリズムはこれらの接続のいくつかを見逃すことになる。  
有利な回復条件であっても、クロスバリデーション（GraphicalLassoCVオブジェクトを使用するなど）で選択されたアルファ・パラメータは、多すぎるエッジを選択することにつながる。ただし、関連するエッジは、関連しないエッジよりも重い重みを持つことになる。

極小サンプル設定における共分散行列と精度行列の最尤推定値、収縮推定値、スパース推定値の比較。  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/aaa000bb-4a3b-586c-735d-12f84d621f69.png)
(図：[2.6.3. Sparse inverse covariance](https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance)より)

学習
```python
%%time
# GraphicalLassoによる構造学習
print(time.ctime()) 
model = GraphicalLassoCV(alphas=list(np.logspace(-1,0,10)), max_iter=200)
model.fit(df_std)

cov=np.cov(df_std.T)  # 計算による相関行列
cov_ = model.covariance_  # スパース化した相関行列
pre_ = model.precision_  # スパース化した相関行列の逆行列(精度行列)
pcm = np.empty_like(pre_)  # スパース化した偏相関行列
for i in range(pre_.shape[0]):
    for j in range(pre_.shape[0]):
        pcm[i, j] = -1*pre_[i, j]/np.sqrt(pre_[i, i]*pre_[j, j])
sparsity = np.mean(pre_==0) * 100
print('Selected alpha', model.alpha_)
print(f'0と推定された成分の割合: {round(sparsity, 3)}%')
print(time.ctime())

# modelを保存するなら実行
save_pickle = 'pickles'
os.makedirs(save_pickle, exist_ok=True)
#with open(save_pickle+'/model_glasso.pickle', mode='wb') as f:
#    pickle.dump(model, f)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/4c3512b5-fd8f-7ad7-dc32-6fef47b093d9.png)

無向グラフで可視化
```python
# 偏相関行列を無向グラフで可視化(networkx使用)
thres = 0.0
adj = np.where(pcm<=thres,0,pcm)
adjacency_ = pd.DataFrame((adj)-np.diag(np.diag(adj)), columns=colsname, index=colsname)
G=nx.from_pandas_adjacency(adjacency_, create_using=nx.Graph)
plt.figure(figsize=(15,10))
#pos = nx.bipartite_layout(G, colsname[:5])
#pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
edge_labels = edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)])#nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx(G, pos)
plt.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/667d993c-c60e-361a-927a-c30808413db1.png)

# 考察
## 想定される因果
連続値変数から異常種類フラグにつながる因果があるはず。
品質タイプは説明文を見ると、摩耗量に関連しているようなので、Tool wearに因果があると思われる。
回転速度からトルクや、回転速度からプロセス温度、気温からプロセス温度にも因果があると想定。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/5897ab76-8ec4-5eae-63ca-2c82da4a0302.png)

## Direct LiNGAM ( & bootstrap)
異常種類フラグから連続値変数への因果を指すことが多く、現実の因果に沿っていない結果となっている。  
 __ただ、出力した因果がすべて逆だったらかなり妥当な結果になる。__  
2値変数を標準化して連続値としているのでそのせいで因果の向きが逆になる現象が起きてしまっているのかもしれない。
ランダム故障RNFが因果なしとなっているのは現実的である。  
bootstrapの推定は毎回結果が変わるが、通常の推定と因果の向きなどは変化はない傾向。

Direct LiNGAM  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/79a967f0-fe89-fa40-c059-782e39a0fbf7.png)
Direct LiNGAM bootstrap  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/3b883eff-487c-fbc4-300a-d35f82369511.png)

## RESIT ( & bootstrap)
非線形の因果を捉えることができるモデルなので、因果のパラメータは因果あり:1、因果なし:0の2値である。
因果のエッジが多く、連続値から異常種類フラグへの因果(TemperatureからHDFなど)もあるが、逆(TWFからTorqueなど)もある。
bootstrapの推定では異常種類フラグから連続値への因果は減る。
想定より因果のエッジが多すぎる。
ランダム故障RNFが様々な変数から因果ありとなっているのは非現実的である。

RESIT  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/1ac60b74-91be-e848-1eee-163eb8ce1327.png)
RESIT bootstrap  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/91e78fc1-c869-888a-710d-74229ae89d2b.png)

## RCD ( & bootstrap)
未観測共通要因がある場合にも対応できるモデル。
同じ未観測共通要因があるもの同士の変数はnanのエッジでつながれている。
ほとんどのエッジがnanでつながれていて、因果関係の発見に失敗していると思われる。
因果ありとなっているエッジも異常種類フラグから連続値への因果が多く現実的でない。
bootstrap推定に関してはすべてのエッジがnanでつながれていて、因果関係の発見に失敗している。
ただ、ランダム故障RNFが因果なしとなっているのは現実的である。

RCD  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/dcab4c34-1901-75a6-d993-fc4fb4b4513b.png)
RCD bootstrap  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d286bbdb-81b4-e46c-2e2a-20d64309234a.png)

## CAM-UV
非線形かつ未観測共通要因がある場合にも対応できるモデル。
因果のパラメータは因果あり:1、因果なし:0の2値である。
同じ未観測共通要因があるもの同士の変数はnanのエッジでつながれている。
因果ありとなっているエッジも異常種類フラグから連続値への因果が多く現実的でない。
ただ、ランダム故障RNFが因果なしとなっているのは現実的である。

CAM-UV  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/917300aa-e7f3-a589-97fc-31a8f3688f31.png)

## NOTEARS
数理最適化でDAGを構築するモデル。
- 正則化項がない場合の標準のNOTEARSについて、今までで最も妥当性が高い。Direct LiNGAMで「因果が逆なら、妥当性があるが…」と述べたが、この結果はその状態に近く、想定される因果に近い。  
- NOTEARS Lassoについて、L1ペナルティを与えると一部異常種類フラグから連続値変数への因果が生じ妥当性が下がる(原因不明)。  
- NOTEARS pytorchで離散変数設定あり、L1ペナルティありの場合、離散変数同士の因果が支配的になり妥当性が下がる。  
- NOTEARS pytorchで離散変数設定なし、L1ペナルティありの場合、標準のNOTEARSから正則化でいくつかのエッジがなくなったようなグラフになり、妥当性が高い。  

標準のNOTEARSを使うか、エッジを減らしたい場合NOTEARS pytorchでL1ペナルティをつけるほうがいいと思われる。

標準のNOTEARS（0.1以下のエッジは削除）  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6a0e748c-07ee-0f34-c987-70e93f43b189.png)
NOTEARS Lasso（0.1以下のエッジは削除、L1ペナルティ0.01）  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/092c9021-47d9-0dd3-164a-dbe44351338a.png)
NOTEARS pytorch(0.1以下のエッジは削除、離散変数設定あり、L1ペナルティ0.1)  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/259f5e78-f392-a542-3f5b-74f47286a96d.png)
NOTEARS pytorch(0.1以下のエッジは削除、連続値や離散値の指定なし、L1ペナルティ0.01)  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7f024a28-880b-9bed-5b32-7c6b9c0c39d0.png)

## Bayesian network
スコアベースや制約に基づく構造学習でDAGを構築し条件付き確率分布（CPD）を用いてパラメータ化するするモデル。  
連続値変数は離散化して構築したが、妥当な因果グラフになっているとはいいがたい。  
エッジも少なく、離散化した連続値変数から異常種類フラグへ因果が伸びているものもある。

Bayesian network  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/4b68a186-fe1b-55d8-3bc5-44af1d6f69cd.png)

## Graphical Lasso
分散共分散行列の逆行列(精度行列)をスパース推定するアルゴリズム。  
精度行列から偏相関行列を求められ、それを構造とする。  
比較的想定に近い構造は得られているが、NOTEARSやDirect LiNGAMと違い、トルクと回転速度の関連が示されない。

Graphical Lasso  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9e9ad642-cf13-d0f4-d60d-2dcac7d01de6.png)

# おわりに
「AI4I 2020 Predictive Maintenance Dataset Data Set」において、想定されるグラフをもっともよく再現できたのはNOTEARSだった。  
Direct LiNGAMはエッジをつなぐ先は想定に近いが、因果の向きが逆になっていることが多く、その部分でNOTEARSに劣っていたが、連続値のみのデータの場合うまくいく可能性もある。
その他非線形や離散変数、未観測共通要因に対応できるモデルについては想定されるような因果グラフはできなかった。  
あくまで今回のデータで実施した感触としては、まず初手としてNOTEARS、もしくはDirect LiNGAMを用いて因果グラフ構築を試みた方が良いと感じた。

以上！
