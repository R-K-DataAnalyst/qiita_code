# はじめに
sklearnで決定木モデルを作った時、sklearn標準の可視化機能で可視化してもわかりにくい。ということでdtreevizを使って可視化することも多いと思う。
[標準の可視化イメージ](https://scikit-learn.org/stable/modules/tree.html)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/bd617642-9c00-f6b1-7c52-530bd4fd2017.png)
[dtreevizの可視化イメージ](https://github.com/parrt/dtreeviz)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9c121a18-46f6-526a-2959-6f7417073509.png)

ただ、個人的にはdtreevizよりもSPSS Modelerのような可視化がしたい…。
[SPSS Modelerの可視化イメージ](https://www.ibm.com/blogs/solutions/jp-ja/spssmodeler-push-node-6/)（※これはChaidの結果）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/75aa4726-2478-a61c-04fa-89cd6fc45024.png)

いろいろ調べていると、Rosyukuさんの「[Scikit-learnで学習した決定木をETEを使って可視化するモジュール（eteview）を構築してみた](https://own-search-and-study.xyz/2017/01/29/scikit-learn%E3%81%A7%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E6%B1%BA%E5%AE%9A%E6%9C%A8%E3%82%92ete%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E5%8F%AF%E8%A6%96%E5%8C%96%E3%81%99%E3%82%8B%E3%83%A2%E3%82%B8/)」がかなりドンピシャな内容だったので、コードを読ませてもらいかなり参考にさせてもらってSPSS Modelerの決定木のような可視化を行ってみた。※ete3はpip install ete3でインストールした(conda installだとimport時にエラーが出た)

こんな感じ↓
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/5116a315-bd29-fb94-b369-b8d0ed5a2955.png)

# データ
KaggleのTelco Customer Churnのデータを使用する。
https://www.kaggle.com/blastchar/telco-customer-churn
これは電話会社の顧客に関するデータであり、Churn(解約)するか否かを目的変数とした2値分類問題。
各行は顧客を表し、各列には顧客の属性が含まれている。

# 決定木モデル作成
欠損行とかもさらっと削除して、カテゴリカルデータはサクッとLabelEncodingも行って前処理は終わらせる。

```python
# パッケージインポート
import os
import re
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy
import seaborn as sns
import gzip
import glob
import datetime as dt
import gc
import sys
import tqdm as tq
from tqdm import tqdm
import time
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import tree
from six import StringIO
import pydotplus
from IPython.display import Image
import ete3
from ete3 import Tree, TreeStyle, TextFace, PieChartFace, BarChartFace

jpn_fonts=list(np.sort([ttf for ttf in fm.findSystemFonts() if 'ipaexg' in ttf or 'msgothic' in ttf or 'japan' in ttf or 'ipafont' in ttf]))
jpn_font=jpn_fonts[0]
prop = fm.FontProperties(fname=jpn_font)
print(jpn_font)
sns.set()

# データ読み込み
churn=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 半角空白をNanに変更
churn.loc[churn['TotalCharges']==' ', 'TotalCharges']=np.nan
# floatに変更
churn['TotalCharges']=churn['TotalCharges'].astype(float)
churn.dropna(inplace=True)
display(churn.head())

# LabelEncodingしたデータフレームchurn_encodeを作る
churn_encode=churn.copy()
columns=list(churn_encode.select_dtypes(include=['object']).columns.to_numpy())
for column in columns:
    le = preprocessing.LabelEncoder()
    le.fit(churn_encode[column])
    churn_encode[column] = le.transform(churn_encode[column])
churn_encode=churn_encode.drop('customerID',axis=1)

# 説明変数名
colx=churn_encode.columns.to_numpy()[:-1]
# 目的変数名
coly='Churn'
dfx=churn_encode[colx].copy()# 説明変数
dfy=churn_encode[coly].copy()# 目的変数
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/37044f75-dbad-1de4-75d9-2bb5581aedae.png)

深さ3で決定木モデルを作り、まず標準の可視化機能で可視化する。

```python
# 決定木モデル構築
clf = sklearn.tree.DecisionTreeClassifier(max_depth=3, random_state=0)
clffit = clf.fit(dfx.to_numpy(), dfy.to_numpy())

# 決定木の可視化
dot_data = StringIO()
sklearn.tree.export_graphviz(clffit, out_file=dot_data
                             , feature_names=colx
                             , class_names=True
                             , filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7e9928db-77f3-be7d-523c-5065ef0ee030.png)
うーん、わかりづらい。

ete3を使って、この決定木の結果を可視化する。
そもそもete(=[ETE Toolkit](http://etetoolkit.org/))は"A Python framework for the analysis and visualization of trees."であり、いろんな樹形図の可視化ができるようだ。Graphvizをインストールしなくても決定木を可視化できるのも地味にうれしい。

まずete3で作るとどのような可視化になるのか最初に見せる。
make_tree_ete3という関数を定義して実行すると、新たにウィンドウが開き、決定木の可視化結果が出力される。

```python
def make_tree_ete3(clffit, tree_cols, obj_unique, fsize=100, height=300, visual=True, name='tmp'):
    # eteのTreeインスタンスを構築
    tree = Tree()
    
    # node size
    # 各ノードのサンプル数の順に木のノードの大きさを変える
    # tree_.n_node_samplesはノードごとのサンプル数を返す
    nodesize={}
    for i, n in enumerate(np.sort(clffit.tree_.n_node_samples)):
        nodesize[n] = i+1
        
    # 各ノードを設定していく
    # tree_.node_countはノード数を返す
    for i in range(clffit.tree_.node_count):
        #i=0、つまりルートノードの名称を0にする
        if i == 0:
            tree.name = str(0)
        
        # 設定するノードを指定
        # name=str(i)であるete3親ノードの設定をする
        node = tree.search_nodes(name=str(i))[0]
        
        # ノードごとにChurn別の%を計算し、配分の円グラフを作成
        #Graph_Object = PieChartFace(percents=clffit.tree_.value[i][0] / clffit.tree_.value[i].sum() * 100# Churn別の割合
        #                            , width=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
        #                            , height=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
        #                            , colors=ete3.COLOR_SCHEMES['set2'])# グラフの色
        # ノードごとにChurn別の数を計算し、barグラフを作成
        Graph_Object = BarChartFace(values=clffit.tree_.value[i][0]# Churn別のサンプル数
                                    , width=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                                    , height=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                                    , colors=ete3.COLOR_SCHEMES['set2'])# グラフの色
        Graph_Object.opacity = 0.8
        #Graph_Object.hz_align = 2# 0 left, 1 center, 2 right
        #Graph_Object.vt_align = 2# 0 left, 1 center, 2 right
        
        #グラフをセット
        # position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
        node.add_face(Graph_Object, column=2, position="branch-right")

        # 左下の子ノードの設定をする
        if clffit.tree_.children_left[i] > -1:# 左下に子ノードがある場合(-1の時、子ノードはない)
            # ノード名称はsklearnのtreeのリストIDと一致させる
            node.add_child(name=str(clffit.tree_.children_left[i]))# ete3子ノード追加
            # 対象を子ノードに移す
            node = tree.search_nodes(name=str(clffit.tree_.children_left[i]))[0]
            # 分岐条件を追加
            # position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
            node.add_face(TextFace(tree_cols[clffit.tree_.feature[i]], fsize=fsize)# 特徴量の名前
                          , column=0, position="branch-top")# Text位置
            node.add_face(TextFace(u"≦" + "{0:.2f}".format(clffit.tree_.threshold[i]), fsize=fsize)# 特徴量の分岐の閾値
                          , column=1, position="branch-bottom")# Text位置
            # 親ノードに対象を戻しておく
            node = tree.search_nodes(name=str(i))[0]
        
        # 右下の子ノードの設定をする
        if clffit.tree_.children_right[i] > -1:# 右下に子ノードがある場合(-1の時、子ノードはない)
            # ノード名称はsklearnのtreeのリストIDと一致させる
            node.add_child(name=str(clffit.tree_.children_right[i]))# ete3子ノード追加
            # 対象を子ノードに移す
            node = tree.search_nodes(name=str(clffit.tree_.children_right[i]))[0]
            # 分岐条件を追加
            # position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
            node.add_face(TextFace(tree_cols[clffit.tree_.feature[i]], fsize=fsize)# 特徴量の名前
                          , column=0, position="branch-top")# Text位置
            node.add_face(TextFace(">" + "{0:.2f}".format(clffit.tree_.threshold[i]), fsize=fsize)# 特徴量の分岐の閾値
                          , column=1, position="branch-bottom")# Text位置
            # 親ノードに対象を戻しておく
            node = tree.search_nodes(name=str(i))[0]

        # ノード内のサンプル数や割合をテキストとして記す
        text1 = str(obj_unique[0])+":{0:.0f}".format(clffit.tree_.value[i][0][0] / clffit.tree_.n_node_samples[i] * 100) + "%"
        text1_1 = "{0:.0f}".format(clffit.tree_.n_node_samples[i])
        texts_ary = []
        for obj_i in obj_unique:# 各クラスのサンプル数
            txt = "{:.0f}:{:.0f}".format(obj_i, clffit.tree_.value[i][0][obj_i])
            texts_ary.append(txt)
        
        # 情報を書き込み
        # position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
        node.add_face(TextFace(text1_1, fsize=fsize)
                      , column=4, position="branch-right")
        node.add_face(TextFace(text1, fsize=fsize)
                      , column=4, position="branch-right")
        for txt_i in texts_ary:
            node.add_face(TextFace(txt_i, fsize=fsize)
                          , column=4, position="branch-right")

    if visual:
        # 別ウィンドウで可視化
        tree.show()#tree.render("%%inline")#, tree_style=ts)
    else:
        #ファイル保存
        tree.render(name+'.svg', h=height)#, tree_style=ts)
```

※追記  
Jupyterで可視化するとき、もしカーネルが〇んでしまうなら`os.environ['QT_QPA_PLATFORM']='offscreen'`を設定するといいかもしれない。（https://github.com/etetoolkit/ete/issues/296）
```python
obj_unique = np.sort(dfy.unique())# 目的変数の種類のリスト（今回はChurnの0と1）
make_tree_ete3(clffit, colx, obj_unique, fsize=50, height=300, visual=True, name='tmp')
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/3cbda40f-8da5-dd9c-bde0-3a8422a42569.png)
SPSS Modelerの可視化にけっこう近づいたように見える。
上図では、各ノードでChurnが0のサンプル数を緑、1のサンプル数を橙でbarグラフ表示している。
グラフの右側の文字が、上から順に合計サンプル数、Churnが0の割合、Churnが0のサンプル数、Churnが1のサンプル数となっている。
分岐の先には分岐の条件である特徴量名とその閾値が書かれている。各ノードの大きさがサンプル数に比例して大きく表示されるようになっている。
一目見て、木の上の分岐に行くとChurnが1の人の割合が高くなることがわかる。そう、見たかったのはこれなんだよ…。
ちなみにBarChartの部分をPieChartに変更すると円グラフに変更できる。

```python
# 関数内のここの部分を
Graph_Object = BarChartFace(values=clffit.tree_.value[i][0]# Churnが0のサンプル数
                            , width=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                            , height=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                            , colors=ete3.COLOR_SCHEMES['set2'])# グラフの色
```
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
```python
# こう変える
Graph_Object = PieChartFace(percents=clffit.tree_.value[i][0] / clffit.tree_.value[i].sum() * 100# Churnが0の割合
                            , width=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                            , height=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                            , colors=ete3.COLOR_SCHEMES['set2'])# グラフの色
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/cb4d1087-638a-36f1-590c-664363461ca8.png)

見やすい…。ete3、ええやん…。(ステキ)
ちなみに、tree.show()でJupyter Notebook上で出力でき、tree.render()でsvgやpngで結果を保存できる。

ではmake_tree_ete3関数の中を見ていってみよう。

まず、Treeインスタンスを構築。このtreeにノードやテキストなど色々つけ足していく。
各ノードのサンプル数に応じてノードの大きさを変えたいのでノードのサンプル数順にランクをつけておく。

```python
# eteのTreeインスタンスを構築
tree = Tree()

# node size
# 各ノードのサンプル数の順に木のノードの大きさを変える
# tree_.n_node_samplesはノードごとのサンプル数を返す
nodesize={}
for i, n in enumerate(np.sort(clffit.tree_.n_node_samples)):
    nodesize[n] = i+1
nodesize
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f77a3201-8b46-2197-6497-e01d1e696f31.png)

treeはforループで各ノードを設定していくが、内容の説明のため、ルートノードの設定だけを書いていく。
各ノードに名前を付けて設定していく。ルートノードは名称を"0"として設定していく。

```python
i=0
fsize=100
#i=0、つまりルートノードの名称を0にする
if i == 0:
    tree.name = str(0)
        
# 設定するノードを指定
# name=str(i)であるete3親ノードの設定をする
node = tree.search_nodes(name=str(i))[0]
```

まず指定したノードにete3のBarChartFaceクラスを加える。
BarChartFaceではclffit.tree_.value[i][0]でChurn別のサンプル数を計算しbarグラフを作成する。
グラフの大きさwidth, heightは最初に定義したnodesizeをもとに設定する。
add_faceでBarChartFaceクラスを加えられる。（columnは"An integer number starting from 0"）
ここまでの状態で可視化してみるとルートノードのbarグラフが描かれている。

```python
# ノードごとにChurn別の数を計算し、barグラフを作成
Graph_Object = BarChartFace(values=clffit.tree_.value[i][0]# Churn別のサンプル数
                            , width=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                            , height=nodesize[clffit.tree_.n_node_samples[i]]*100# nodesize辞書内の数値×100
                            , colors=ete3.COLOR_SCHEMES['set2'])# グラフの色
Graph_Object.opacity = 0.8
#Graph_Object.hz_align = 1# 0 left, 1 center, 2 right
#Graph_Object.vt_align = 1# 0 left, 1 center, 2 right
# グラフをセット
# position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
node.add_face(Graph_Object, column=2, position="branch-right")
tree.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/2a2ecf20-dd5f-d54a-8d9a-30dc5d2d1215.png)

次に分岐条件に関することをテキストで記入していくが、今まではルートノードの設定をしていたが、分岐条件などはルートノードの子ノード側の設定として書いていく。
親ノードであるルートノードの下の左右の子ノードはclffit.tree_.children_leftとclffit.tree_.children_rightで取得できる。
ノードの設定対象を子ノードに移し、特徴量や分岐条件をclffit.tree_.feature, clffit.tree_.thresholdで書いていく。
ただし、clffit.tree_.children_leftとclffit.tree_.children_rightの数値が-1の時、親ノードはtreeの終端で子ノードは無いためこの処理は省略される。
add_faceでテキストを記入できたら、ノードの設定対象を親ノードに戻しておく。
ここまでの状態で可視化してみるとルートノードのbarグラフが描かれており、子ノード側の分岐先に特徴量名と、分岐条件が書かれている。

```python
# 左下の子ノードの設定をする
if clffit.tree_.children_left[i] > -1:# 左下に子ノードがある場合(-1の時、子ノードはない)
    # ノード名称はsklearnのtreeのリストIDと一致させる
    node.add_child(name=str(clffit.tree_.children_left[i]))# ete3子ノード追加
    # 対象を子ノードに移す
    node = tree.search_nodes(name=str(clffit.tree_.children_left[i]))[0]
    # 分岐条件を追加
    # position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
    node.add_face(TextFace(colx[clffit.tree_.feature[i]], fsize=fsize)# 特徴量の名前
                  , column=0, position="branch-top")# Text位置
    node.add_face(TextFace(u"≦" + "{0:.2f}".format(clffit.tree_.threshold[i]), fsize=fsize)# 特徴量の分岐の閾値
                  , column=1, position="branch-bottom")# Text位置
    # 親ノードに対象を戻しておく
    node = tree.search_nodes(name=str(i))[0]
    
# 右下の子ノードの設定をする
if clffit.tree_.children_right[i] > -1:# 右下に子ノードがある場合(-1の時、子ノードはない)
    # ノード名称はsklearnのtreeのリストIDと一致させる
    node.add_child(name=str(clffit.tree_.children_right[i]))# ete3子ノード追加
    # 対象を子ノードに移す
    node = tree.search_nodes(name=str(clffit.tree_.children_right[i]))[0]
    # 分岐条件を追加
    # position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
    node.add_face(TextFace(colx[clffit.tree_.feature[i]], fsize=fsize)# 特徴量の名前
                  , column=0, position="branch-top")# Text位置
    node.add_face(TextFace(">" + "{0:.2f}".format(clffit.tree_.threshold[i]), fsize=fsize)# 特徴量の分岐の閾値
                  , column=1, position="branch-bottom")# Text位置
    # 親ノードに対象を戻しておく
    node = tree.search_nodes(name=str(i))[0]
    
tree.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/bc5fb5ae-b96f-00f5-9f5f-db95eee96e56.png)

最後に、ノード内のサンプル数や割合をテキストとして記入するが、これらは親ノードに対して設定していく。
ノード内のサンプル数、Churnが0のサンプルの割合、Churnが0のサンプル数、Churnが1のサンプルを定義し、add_faceでテキストを記入。
ここまでの状態で可視化してみるとルートノードのbarグラフが描かれており、barグラフの右側にサンプル数などがあり、子ノード側の分岐先に特徴量名と、分岐条件が書かれている。

```python
# ノード内のサンプル数や割合をテキストとして記す
text1 = str(obj_unique[0])+":{0:.0f}".format(clffit.tree_.value[i][0][0] / clffit.tree_.n_node_samples[i] * 100) + "%"
text1_1 = "{0:.0f}".format(clffit.tree_.n_node_samples[i])
texts_ary = []
for obj_i in obj_unique:# 各クラスのサンプル数
    txt = "{:.0f}:{:.0f}".format(obj_i, clffit.tree_.value[i][0][obj_i])
    texts_ary.append(txt)
        
# 情報を書き込み
# position='aligned', 'branch-top', 'float-behind', 'branch-bottom', 'float', 'branch-right'
node.add_face(TextFace(text1_1, fsize=fsize)
              , column=4, position="branch-right")
node.add_face(TextFace(text1, fsize=fsize)
              , column=4, position="branch-right")
for txt_i in texts_ary:
    node.add_face(TextFace(txt_i, fsize=fsize)
                  , column=4, position="branch-right")

tree.show()
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/5fa36b0f-b51f-384a-0c82-9d495d39de3e.png)

これをforループですべてのノードに対して処理を行うと、決定木の可視化が完成する。
（再掲）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/5116a315-bd29-fb94-b369-b8d0ed5a2955.png)
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/cb4d1087-638a-36f1-590c-664363461ca8.png)
標準機能と比較すると見やすいと思う。（個人的な感想）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7e9928db-77f3-be7d-523c-5065ef0ee030.png)

# おわりに
ete3を使ってsklearnの決定木(CART)可視化を実施した。
SPSS Modelerの図に寄せることを目標にしたが、なかなかいい感じにできたと思う。
分類木の可視化ならdtreevizのものよりも個人的に好みなものができた。
カスタマイズもいろいろできそうだし回帰木の可視化も作ろうと思えばいろいろ作れると思う。Graphvizのインストールがいらないのも個人的にGood。
ちなみに決定木の向きを縦に変更できるのかはわからない。

以上！

# おまけ：sklearnのclf.fit後のclf.tree_.~~について
>clf.tree_.n_node_samples

過去ノードのサンプル数
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/41b021d7-4f0d-52e3-7cfa-e037e40eabb0.png)

>clf.tree_.node_count

ノードの数
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/2a3a5848-5b68-2cfc-6776-69dfd5b78ac6.png)

>clf.tree_.value

各ノード、各クラス(Churn 0 or 1)のサンプル数
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/c0e78caf-1b19-c6ab-4e7d-ae26ae1e5ffa.png)

>clf.tree_.children_left

各ノードの左下の子ノード
例：ノード0の左下の子ノードはノード1、ノード2の左下の子ノードはノード3、ノード3の左下の子ノードは無い
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/fd50d04e-36e4-874e-bf25-92ee889c3f7a.png)

>clf.tree_.feature

各ノードの分岐条件の特徴量のIndex番号
例：ノード0の分岐条件の特徴量は、特徴量リストの14番目(='Contract')
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/8d7501a0-ba57-7745-dd8e-a5e1975fc861.png)

>clf.tree_.threshold

各ノードの分岐条件（「〇〇超過、〇〇未満」の〇〇に該当）
例：ノード0は'Contract'は0.5を閾値に分岐
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6b8acaa1-62f5-3a93-f5f5-d2c803e46ff0.png)
