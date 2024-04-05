# はじめに
以前、こんな記事を書いた。
* [傾向スコアマッチング(python)](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)

pythonで傾向スコアマッチングするってだけの記事。
そして最後にこう書いた。
>実は知らないだけでサクッとやってくれるライブラリとかあるんだろうか。
あったらそれ使いたいなーと思った。

はい、ありました。
- [DoWhy](https://www.pywhy.org/dowhy/v0.9.1/)
- [EconML](https://econml.azurewebsites.net/index.html)

どちらもMicrosoftがリリースした因果推論ライブラリ。

DoWhyが基本的な因果推論をするためのライブラリで、「効果推定、因果構造学習、因果構造の診断、根本原因分析、介入、反実仮想のための様々なアルゴリズムを提供」しているそう。
DoWhyを使うと、傾向スコアマッチングもできた。

EconMLは、機械学習技術を応用して、因果を推定するライブラリ。条件付き平均処置効果(CATE)を推定できる。Meta-learnerとかのあれ。
そのあたりの詳細は、書籍[「つくりながら学ぶ! Pythonによる因果分析 ~因果推論・因果探索の実践入門~」](https://www.amazon.co.jp/dp/4839973571)を参照。

今回は、以前書いた記事「[傾向スコアマッチング(python)](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)」と同様に、Lalondeデータセットの因果推論をDoWhy(& EconML)を使ってやってみる。
EconMLは単体で使うこともできるが、今回はDoWhy経由でEconMLを使ってみる。（後述しているがDoWhy経由でEconMLを使うとおかしな現象が起こってしまったので、単体で使ったほうがいいかも。）

# 参考
- [DoWhy documentation](https://www.pywhy.org/dowhy/v0.9.1/)
- [DoWhy github](https://github.com/py-why/dowhy)
- [EconML documentation](https://econml.azurewebsites.net/index.html)
- [EconML github](https://github.com/py-why/EconML)
- [DoWhyとEconMLによる因果推論の実装](https://speakerdeck.com/s1ok69oo/dowhytoeconmlniyoruyin-guo-tui-lun-noshi-zhuang)
- [統計的因果推論のためのPythonライブラリDoWhyについて解説：なにができて、なにに注意すべきか](https://www.krsk-phs.com/entry/2018/08/22/060844)
- [機械学習で因果推論~Meta-LearnerとEconML~](https://zenn.dev/s1ok69oo/articles/1eeebe75842a50)
- [「つくりながら学ぶ! Pythonによる因果分析 ~因果推論・因果探索の実践入門~」](https://www.amazon.co.jp/dp/4839973571)
- [効果検証入門～正しい比較のための因果推論/計量経済学の基礎～](https://www.amazon.co.jp/dp/4297111179)
- [LaLonde(1986)とその周辺の学習記録](https://moratoriamuo.hatenablog.com/entry/2020/02/10/235636)
- [ “因果推論駅”の奥の方を探訪しながら考える：われわれの諸研究は内的に/外的にどのような繋がりを持っているのか](https://speakerdeck.com/takehikoihayashi/wai-de-nidofalseyounaxi-gariwochi-tuteirufalseka)

# データ
「[傾向スコアマッチング(python)](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)」と同様。
効果検証入門の書籍でも使われているLaLonde(1986)のデータ。
詳細内容は「[LaLonde(1986)とその周辺の学習記録](https://moratoriamuo.hatenablog.com/entry/2020/02/10/235636)」を参照。
NSWのデータはRCTによる結果。
CPSのデータは実験の外で得られた結果。
NSWの一部をCPSに置き換えることで、セレクションバイアスが生じているデータセットを作れる。
https://users.nber.org/~rdehejia/data からpandasでloadして使用。

# データ読み込み
```python
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.linear_model import LassoCV, HuberRegressor, LinearRegression, LogisticRegression
from tqdm import tqdm
import statsmodels.api as sma
import dowhy
import econml
from econml.metalearners import TLearner
from econml.metalearners import SLearner
from econml.dr import DRLearner
from lightgbm import LGBMRegressor
from IPython.display import Image, display
jpn_fonts=list(np.sort([ttf for ttf in fm.findSystemFonts() if 'msgothic' in ttf]))
jpn_font=jpn_fonts[0]
prop = fm.FontProperties(fname=jpn_font)
sns.set()

df_cps1 = pd.read_stata('https://users.nber.org/~rdehejia/data/cps_controls.dta')
df_nsw = pd.read_stata('https://users.nber.org/~rdehejia/data/nsw_dw.dta')
display(df_cps1.head())
display(df_nsw.head())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ec01aef4-e390-5feb-5dcc-9b55642d8932.png)

# セレクションバイアスがあるデータ作成
「[傾向スコアマッチング(python)](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)」と同様。
定義したdf_nswの対照群('treat'=0)をdf_cps1のデータに入れ替える。
以下のような不均衡なデータになる。
- 処置群：185人
- 対照群：15,992人
```python
# df_nswの対照群をdf_cps1に入れ替える
tr_col = 'treat'  # 介入変数
target_col = 're78'  # 目的変数
all_col = ['age','education','black','hispanic','married','nodegree','re74','re75','re78']
exp_col = ['age','education','black','hispanic','married','nodegree','re74','re75'] # 共変量
num_col = ['age','education','re74','re75']
cat_col = ['black','hispanic','married','nodegree']
df_nswcps1 = pd.concat([df_nsw[df_nsw[tr_col] == 1], df_cps1], ignore_index=False).reset_index(drop=True)
display(df_nswcps1)
display(df_nswcps1.treat.value_counts().reset_index())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d42856f4-c979-239c-4213-dfe354000657.png)

# Lalondeデータセットの本当の効果の確認
置き換える前のNSWのデータはRCTによる結果なので、このデータで回帰分析をすると本当の介入効果がわかる。
「[傾向スコアマッチング(python)](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)」では、statsmodelで回帰分析をしたが、今回はDoWhyを使って実施してみる。
DoWhyでは以下のように記述して回帰分析ができる。
're78'への介入効果は$1,671であり統計的にも有意。
DoWhyとEconMLを使ってこの介入効果を推定できるか実践していく。
```python
### 本来の効果を回帰分析で求める ###
lalonde = df_nsw.drop(columns=['data_id']).copy()  # idを抜いたデータフレーム
lalonde[tr_col] = lalonde[tr_col].astype(bool)  # dowhyでは2値の介入変数はbool型に変換する
# モデルの定義(回帰モデル)
model=dowhy.CausalModel(data = lalonde  # データフレーム
                        , treatment='treat'  # 介入変数カラム名
                        , outcome='re78'  # 目的変数カラム名
                        , common_causes='nodegree+black+hispanic+age+education+married'.split('+')  # 共変量
                       )
# 因果効果を推定
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand
                                 , method_name='backdoor.linear_regression'
                                 , test_significance=True
                                 , confidence_intervals=True
                                )
print(estimate)
print('Causal Estimate is ' + str(estimate.value))
print('Confidence Intervals', estimate.get_confidence_intervals())
print('p-value', estimate.test_stat_significance())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/301efb90-facf-19af-6d56-93fe4c90b8a0.png)

# 前処理
因果推論用のデータフレームを作成する。
```python
### 因果推論用のdf作成 ###
X_train = df_nswcps1[exp_col].copy()  # 共変量
y_train = df_nswcps1[tr_col].copy()  # 介入変数
target = df_nswcps1[target_col].copy()  # 目的変数
# 共変量の標準化
sc = sklearn.preprocessing.StandardScaler()  # 標準化
X_train_std = pd.DataFrame(sc.fit_transform(X_train), columns=exp_col)  # 標準化
target_std = target.copy()
display(X_train_std)  # 標準化共変量
dataset_init = pd.concat([X_train_std, target_std, y_train], axis=1)
dataset_init[tr_col] = dataset_init[tr_col].astype(bool)
display(dataset_init)  # 作成したデータフレーム
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/aa3a5edb-d529-7608-1cb0-ba31d4c34989.png)


# DAGの定義
介入変数、目的変数、共変量などがどのような関係性になっているか、DAG(Directed Acyclic Graph / 有向非巡回グラフ)をしっかり考えて定義してあげないとDoWhyは使えない。
逆に言うと、なんとなく頭の中で各データの関係性を定義していたものをしっかり書き出す必要があるので、その仮定が本当に成り立つのか、おかしな関係性になっていないかということを考えたりしやすいと思う。
[こちらのリンク](https://www.krsk-phs.com/entry/2018/08/22/060844)では、
>DoWhyではDAGを自分で書いて、読み込ませる必要があります。
あくまでDAGは自分の知識を使って書かないといけないわけで、完璧なDAGが書けるとも限りません。
すなわち、DoWhyを使えばコンピュータが勝手に因果推論を行ってくれるなんて代物ではありません。これはとても重要なポイントです。人間の頭を使う必要があります。だからこそ仮定（Assumption)なのであって、その仮定が成り立たない場合になにが起こりうるのかをあとで検討（Sensitivity Analysis)を行う必要があります。

と書かれており、まさにその通りだなと思う。

[公式のexample：load_graph_example](https://www.pywhy.org/dowhy/v0.9.1/example_notebooks/load_graph_example.html)によるとDAGは"GML graph format"か"DOT format"で定義するとのこと。
グラフは文字列として読み込むか、ファイル(拡張子'gml'または'dot'）として読み込むかのどちらか。
ということでDOT formatで文字列としてDAGを定義。
DOT formatでの書き方は[公式のexample：dowhy_example_effect_of_memberrewards_program](https://www.pywhy.org/dowhy/v0.9.1/example_notebooks/dowhy_example_effect_of_memberrewards_program.html)を参考にした。
"DAGをしっかり考えて定義"とか書いたばっかりだけどしっかりは考えていない。(てへぺろ)
```python
# 介入変数、目的変数、共変量のDAG
causal_graph = \
"""
digraph {
age[label="Age"];
education[label="Education"];
black[label="Black"];
hispanic[label="Hispanic"];
married[label="Married"];
nodegree[label="No degree"];
re74[label="re74"];
re75[label="re75"];
re78[label="re78_target"];
treat[label="treat"];
age -> {treat re78};
education -> {treat re78};
black -> {treat re78};
hispanic -> {treat re78};
married -> {treat re78};
nodegree -> {treat re78};
re74 -> {treat re78};
re75 -> {treat re78};
treat -> re78;
}
"""
```
一度DoWhyを使って、この定義したDAGを可視化する。`view_model`で可能。
```python
# データ
dataset = dataset_init.copy()
#display(dataset.head(1))
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
# DAG可視化
model.view_model()
display(Image(filename="causal_model.png"))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/0342cb7e-4649-0d97-c1fd-676ab0a6b4fc.png)
共変量は介入変数と目的変数に矢印が伸びており、介入変数からは目的変数に伸びている。つまり、介入変数と目的変数に影響を与えるバックドアパスがある状態ということを想定している。
このままだと適切な介入効果を計算できないため、因果推論のテクニックを使ってバックドアパスを閉じてセレクションバイアスの影響を消す必要がある。

ここでバックドアパスがある状態についてもう少し身近な具体例を挙げてみると、

`お題：ECサイトで配布されたクーポンを使った(介入された)顧客の将来の購入金額(目的変数)が上がったかどうか確認したい`

`バイアス：クーポンを使っていないグループより使ったグループの方が、日常的にサイトを利用しているような優良顧客の割合が高い可能性がある(日常的にサイトを使っている優良ユーザーはそりゃクーポン使うよね)`

`バックドアパス：優良顧客かどうかを決める過去の購買ログや顧客属性情報など(共変量)が、クーポンを使ったかどうかに影響しており、また将来の購入金額が高いかどうかに影響している`

上記の設定の場合、そのままクーポンを使ったグループと使っていないグループの購入金額を比較すると、優良顧客の割合が高いことによりクーポンを使っているほど将来の購入金額は高いというクーポンの効果について何の検証にもなっていない当たり前の結果が出てしまう。
そのため、例えば傾向スコアマッチングや層別分析をすると、「共変量(=購買傾向)は似ているがクーポンを使ったグループと使っていないグループ」を比較することができ、クーポンを使ったことによる本当の効果を推定することができる。
これがバックドアパスを閉じてセレクションバイアスの影響を消すということである。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/3ce497e7-a2cd-a1ea-c590-71b01e58cb5d.png)

今回のデータセットでも上記具体例のようなDAGを想定している。

# DoWhyを使った傾向スコアマッチング
DAGさえ考えてしまえばあとはDoWhyで簡単に因果推論の実行はできる。
因果推論の前にまずはセレクションバイアスがある状態ではどういう結果が出るか単純に回帰分析をしてみる。
DAGをもとにモデル定義し、`method_name="backdoor.linear_regression"`で回帰分析を実施。
セレクションバイアスにより介入効果は$699となり、適切な効果測定ができていない。回帰分析の係数も有意ではない。この結果は[前回の記事](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)の時と一致。
```python
### バイアスがある場合の回帰分析の結果 ###
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# 回帰分析
causal_estimate_linear_ate = model.estimate_effect(identified_estimand
                                                   , method_name="backdoor.linear_regression"
                                                   #, target_units="ate"
                                                   , test_significance=True
                                                   , confidence_intervals=True
                                                  )
print(causal_estimate_linear_ate)
print("Causal Estimate is " + str(causal_estimate_linear_ate.value))
print('Confidence Intervals', causal_estimate_linear_ate.get_confidence_intervals())
print('p-value', causal_estimate_linear_ate.test_stat_significance())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f0b73bd4-60cd-f16d-32b5-c14a5ed67523.png)

本来は$1,671程度の効果のはずなので、傾向スコアマッチングをして推定できるか検証。
[前回の記事](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)では自分で実装したが、DoWhyでは回帰分析の時に指定していた`method_name='backdoor.linear_regression'`の部分を`method_name="backdoor.propensity_score_matching"`にすると傾向スコアマッチングを実施できる(他にも引数に違いはあるが)。
傾向スコアを求めるモデルはロジスティック回帰モデルとして、以下のコードを実施すると、結果が出力される。
効果は$1,494となり本来の効果に近くなったことが確認できる。
```python
### 傾向スコアマッチング ###
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# 傾向スコアのためのモデル
lr = LogisticRegression(max_iter=10000, C=500) # ロジスティック回帰
# 傾向スコアマッチング
p_score_matching_att = model.estimate_effect(identified_estimand
                                             , method_name="backdoor.propensity_score_matching"  # 傾向スコアマッチング指定
                                             , target_units='att'  # ATTを推定
                                             , method_params={'propensity_score_model':lr}  # 傾向スコアのためのモデル
                                             #, confidence_intervals=True  # 時間かかる
                                            )

#print('Causal Estimate is ' + str(p_score_matching_att.value))
#print('Confidence Intervals', p_score_matching_att.get_confidence_intervals())
#print('p-value', p_score_matching_att.test_stat_significance())
print(p_score_matching_att)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/a5f4a7e8-7a15-6310-4539-faac82b2be7b.png)

自分で実装した[前回](https://qiita.com/chicken_data_analyst/items/7e1d231ad0ada4ffda8d)の結果では$1,704と異なったが、[githubのコード](https://github.com/py-why/dowhy/blob/8fb32a7bf617c1a64a2f8b61ed7a4a50ccaf8d8c/dowhy/causal_estimators/propensity_score_matching_estimator.py)を見ると、DoWhyでは重複マッチングを許しており、自分で実装したときは重複はさせなかったのでそこで違いが出ていると思われる。またキャリパーの設定も無いっぽいので、一致しなかったと思っている。

ということで傾向スコアマッチングはこんな感じでできる。自分で実装するよりは楽かも。

# DoWhyを使った他のアプローチ
__Distance matching__
距離の指標を定義し、その指標を用いて介入群と非介入群の間の最も近い点をマッチング。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
causal_estimate_dmatch_att = model.estimate_effect(identified_estimand
                                                   , method_name="backdoor.distance_matching"
                                                   , target_units="att"
                                                   , method_params={'distance_metric':"minkowski"}
                                                  )
# distance_metric - 使用する距離メトリック。デフォルトは "minkowski "で、p=2のユークリッド距離計に対応します。

print(causal_estimate_dmatch_att)
print("Causal Estimate is " + str(causal_estimate_dmatch_att.value))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/3d83649b-e85d-d017-91a4-5da289387bcf.png)

__Propensity score stratification__
傾向スコアを使用してデータを層別化。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# 傾向スコアのためのモデル
lr = LogisticRegression(max_iter=10000, C=500) # ロジスティック回帰
causal_estimate_strat_att = model.estimate_effect(identified_estimand
                                                   , method_name="backdoor.propensity_score_stratification"
                                                   , target_units="att"
                                                   , method_params={'num_strata':300
                                                                    , 'clipping_threshold':5
                                                                    , 'propensity_score_model':lr}
                                                  )
# num_strata - データを層別化するビンの数．デフォルトは自動的に決定されます．
# clipping_threshold - 層ごとの処置または対照単位の最小数．初期値=10

print(causal_estimate_strat_att)
print("Causal Estimate is " + str(causal_estimate_strat_att.value))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/4ad7da39-8b8e-8c4a-cf04-c995ec516d92.png)

__Propensity score weighting (IPW)__
データに重みをつけるために（逆）傾向スコアを使用。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# 傾向スコアのためのモデル
lr = LogisticRegression(max_iter=10000, C=500) # ロジスティック回帰
causal_estimate_ips_weight_ate = model.estimate_effect(identified_estimand
                                                       , method_name="backdoor.propensity_score_weighting"
                                                       , target_units = "ate"
                                                       , method_params={"weighting_scheme":"ips_weight"
                                                                        , 'propensity_score_model':lr}
                                                      )
print(causal_estimate_ips_weight_ate)
print("Causal Estimate is " + str(causal_estimate_ips_weight_ate.value))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/96c98cf8-2e0e-3505-4eda-b23f09cfa1fb.png)

IPWだけ効果がマイナスになった。
これは[「効果検証入門～正しい比較のための因果推論/計量経済学の基礎～」](https://www.amazon.co.jp/dp/4297111179)でも書いているように、介入群と非介入群の傾向の違いが大きい場合にIPWは信頼しにくいことが原因。
傾向スコアの逆数をサンプルの重みに利用する特性上、傾向スコアが非常に小さい値をとると、そのサンプルの重みが非常に大きくなるので過剰に水増しされてしまう。
なので、IPWだけ効果がマイナスになったが、ある意味妥当な振る舞い。
マッチング系のアプローチでは効果がプラスになるのでセレクションバイアスを与えたLalondeのようなデータにはそういったアプローチが適切だったということである。

# DoWhy経由でEconMLを使ってCATEの推定
DoWhy経由でEconMLを使ってMeta-learnerによる因果推論をやってみる。
`estimate_effect`関数内の`method_name`の指定を、例えば`"backdoor.econml.metalearners.TLearner"`のようにするとEconMLを使ったCATEの推定ができる。

__T-Learner__
`estimate_effect`関数内の`init_params`で指定できる引数はEconMLのもの。
[ソースコード](https://github.com/py-why/EconML/blob/main/econml/metalearners/_metalearners.py)を見ると、*models*と*categories*を指定できる。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7bbf3aa7-ab6d-3236-860a-eb3dcdac80ed.png)
scikit-learnのAPIがあるモデルなら*models*に渡せるので、今回はLightGBMを渡す。
`estimate_effect`を実施した後のobjectの`cate_estimates`でCATEの推定結果を取り出せる。
`estimate_effect`を実施した後のobjectの`params`内の'*_estimator_object*'からEconMLのobjectを取り出すこともできる。
`target_units`に条件を与えてあげるとその条件に当てはまるサンプルのCATEが出力される(今回は指定無し)。
今回知りたいのは介入したグループの効果なので、*treat*=1のサンプルのCATEの平均がそれにあたる。
結果、効果は$1,771であり、妥当な結果を推定できていると思われる。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# TLearnerで使用するモデル（LightGBM）
clf0 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
clf1 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
# 学習&推論
TLearner_att = model.estimate_effect(identified_estimand
                                     , method_name="backdoor.econml.metalearners.TLearner"  # EconMLが使える
                                     #, target_units=lambda df: df[tr_col]>0  # CATE(treat=1)  # CATEの条件指定可
                                     , method_params={'init_params':{'models':[clf0, clf1]}
                                                      , 'fit_params':{}
                                                     }
                                    )

# 学習させたLightGBMモデル2つ(リスト)
learner = TLearner_att.params['_estimator_object']  # EconML TLearner Object
clf0 = learner.models[0]  # treat=0の学習済みモデル
clf1 = learner.models[1]  # treat=1の学習済みモデル

# ATEの計算
print('ATE', learner.ate(dataset[exp_col]))  #他の求め方：print('ATE', learner.effect(dataset[exp_col]).mean())  #print('ATE', np.mean(clf1.predict(dataset[exp_col].to_numpy()) - clf0.predict(dataset[exp_col].to_numpy())))
# 各サンプルのCATEの平均
print('CATE-mean(All record)', TLearner_att.cate_estimates.mean())
# 介入した各サンプルのCATEの平均
print('CATE-mean(treat=1)', TLearner_att.cate_estimates[dataset[dataset[tr_col]>0].index].mean())
# 介入していない各サンプルのCATEの平均
print('CATE-mean(treat=0)', TLearner_att.cate_estimates[dataset[dataset[tr_col]==0].index].mean())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/a917012b-9ce5-3de3-ae0f-bd28b47baa06.png)


__S-Learner__
`estimate_effect`関数内の`init_params`で指定できる引数はEconMLのもの。
[ソースコード](https://github.com/py-why/EconML/blob/main/econml/metalearners/_metalearners.py)を見ると、*overall_model*と*categories*を指定できる。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7f9645d0-5c1b-6366-ad87-bfc7be3279bf.png)
scikit-learnのAPIがあるモデルなら*overall_model*に渡せるので、今回はLightGBMを渡す。
`estimate_effect`を実施した後のobjectの`cate_estimates`でCATEの推定結果を取り出せる。
`estimate_effect`を実施した後のobjectの`params`内の'*_estimator_object*'からEconMLのobjectを取り出すこともできる。
`target_units`に条件を与えてあげるとその条件に当てはまるサンプルのCATEが出力される(今回は指定無し)。
今回知りたいのは介入したグループの効果なので、*treat*=1のサンプルのCATEの平均がそれにあたる。
結果、効果は$1,877であり、妥当な結果を推定できていると思われる。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# TLearnerで使用するモデル（LightGBM）
clf = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
SLearner_att = model.estimate_effect(identified_estimand
                                     , method_name="backdoor.econml.metalearners.SLearner"
                                     #, target_units=lambda df: df[tr_col]>0  # CATE(treat=1)  # CATEの条件指定可
                                     , method_params={'init_params':{'overall_model':clf}
                                                      , 'fit_params':{}
                                                     }
                                    )

learner = SLearner_att.params['_estimator_object']  # EconML TLearner Object
clf = learner.overall_model  # 学習済みモデル

print('ATE', learner.ate(dataset[exp_col]))
print('CATE-mean(All record)', SLearner_att.cate_estimates.mean())
print('CATE-mean(treat=1)', SLearner_att.cate_estimates[dataset[dataset[tr_col]>0].index].mean())
print('CATE-mean(treat=0)', SLearner_att.cate_estimates[dataset[dataset[tr_col]==0].index].mean())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/dd69bce8-3ac9-9fe4-aedf-c43eca8cabe1.png)

ということでMeta-learnerもわりと使える感じ(X-Learnerは省略)。
ただDRLearnerについては、ver.0.9.1現在DoWhy経由で実施すると[CATEが1サンプルしか出力されないbug](https://github.com/py-why/dowhy/issues/890)があるようなので、DoWhy経由ではなく直接EconMLを使用した方が良さそう。
また、後述するが、他にもDoWhy経由でEconMLを実施する問題点というか疑問点がある。

# DoWhy経由でEconMLを使って生じた疑問点
**1. 直接EconMLを実施したときと微妙に結果が異なる**
直接EconMLでT-Learnerをやってみる。
```python
### 直接EconMLでT-Learnerをやってみる ###
# データ
dataset = dataset_init.copy()
# TLearnerで使用するモデル（LightGBM）
clf0 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)  # treat=0のモデル
clf1 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)  # treat=1のモデル
est = TLearner(models=[clf0, clf1])  # EconML TLearner Object
# 学習
est.fit(dataset[target_col], dataset[tr_col], X=dataset[exp_col])

clf0 = est.models[0]  # treat=0の学習済みモデル
clf1 = est.models[1]  # treat=1の学習済みモデル
# treat=1のサンプル
df1 = dataset[dataset[tr_col]==1].reset_index(drop=True).copy()
# treat=0のサンプル
df0 = dataset[dataset[tr_col]==0].reset_index(drop=True).copy()

print('CATE-mean(All record)', est.effect(dataset[exp_col]).mean())
print('CATE-mean(treat=1)', est.effect(dataset[exp_col])[dataset[dataset[tr_col]>0].index].mean())
print('CATE-mean(treat=0)', est.effect(dataset[exp_col])[dataset[dataset[tr_col]==0].index].mean())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7b425a16-4d88-70f1-119b-69cadc4e7478.png)
DoWhy経由のときは介入効果は1,771だったが、直接EconMLを使用した場合1,976になっていて微妙に一致しない。
LGBMRegressorのハイパーパラメータも同じだし、random_state=0で固定しているのになぜ？

ちなみにS-Learnerは、DoWhy経由でも直接EconMLでも一致する(介入効果1,877)。なぜ？
```python
### 直接EconMLでS-Learnerをやってみる ###
# データ
dataset = dataset_init.copy()
# SLearnerで使用するモデル（LightGBM）
clf = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)  # モデル
est = SLearner(overall_model=clf)  # EconML SLearner Object
# 学習
est.fit(dataset[target_col], dataset[tr_col], X=dataset[exp_col])

clf = est.overall_model  # 学習済みモデル

print('CATE-mean(All record)', est.effect(dataset[exp_col]).mean())
print('CATE-mean(treat=1)', est.effect(dataset[exp_col])[dataset[dataset[tr_col]>0].index].mean())
print('CATE-mean(treat=0)', est.effect(dataset[exp_col])[dataset[dataset[tr_col]==0].index].mean())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/7a1e9181-22b8-9116-518f-6b81decec5b8.png)

**2. 学習済みのモデルの出力がおかしい**
**1.** と関連があるかもしれないがDoWhy経由で学習させたモデルの出力と、直接EconMLで学習させたモデルの出力が異なる。

まず、DoWhy経由でT-Learnerを実施して、学習済みモデルclf0とclf1を取り出す。treat=1のサンプルのみ抽出したdf1も定義しておく。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# TLearnerで使用するモデル（LightGBM）
clf0 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
clf1 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
# 学習&推論
TLearner_att = model.estimate_effect(identified_estimand
                                     , method_name="backdoor.econml.metalearners.TLearner"  # EconMLが使える
                                     #, target_units=lambda df: df[tr_col]>0  # CATE(treat=1)  # CATEの条件指定可
                                     , method_params={'init_params':{'models':[clf0, clf1]}
                                                      , 'fit_params':{}
                                                     }
                                    )
# 学習させたLightGBMモデル2つ(リスト)
learner = TLearner_att.params['_estimator_object']  # EconML TLearner Object
clf0 = learner.models[0]  # treat=0のモデル
clf1 = learner.models[1]  # treat=1のモデル

df1 = dataset[dataset[tr_col]==1].reset_index(drop=True).copy()  # treat=1のサンプル
df0 = dataset[dataset[tr_col]==0].reset_index(drop=True).copy()  # treat=0のサンプル
```

このモデルで実測と予測の散布図yyplotを可視化。
ついでに普通に予測モデルを作った場合の結果も可視化。
明らかにDoWhy経由で学習させたモデルの精度が悪い。
```python
### 実測予測プロット ###
# DoWhy経由で実施すると明らかにmodelの精度が悪い
fig=plt.figure(figsize=(5,5))
plt.scatter(df1[target_col].to_numpy(), clf1.predict(df1[exp_col].to_numpy()))
plt.xlim(0, max(df1[target_col].to_numpy().max(), clf1.predict(df1[exp_col].to_numpy()).max()))
plt.ylim(0, max(df1[target_col].to_numpy().max(), clf1.predict(df1[exp_col].to_numpy()).max()))
plt.xlabel('actual (outcome of treat=1)', fontsize=10)
plt.ylabel('predict (outcome of treat=1)', fontsize=10)
plt.title(learner.__class__.__name__+': DoWhy Train Result', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()


# 普通に予測モデルを学習させると精度はいい
clf0 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
clf1 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
clf0.fit(df0[exp_col], df0[target_col])
clf1.fit(df1[exp_col], df1[target_col])
fig=plt.figure(figsize=(5,5))
plt.scatter(df1[target_col].to_numpy(), clf1.predict(df1[exp_col].to_numpy()))
plt.xlim(0, max(df1[target_col].to_numpy().max(), clf1.predict(df1[exp_col].to_numpy()).max()))
plt.ylim(0, max(df1[target_col].to_numpy().max(), clf1.predict(df1[exp_col].to_numpy()).max()))
plt.xlabel('actual (outcome of treat=1)', fontsize=10)
plt.ylabel('predict (outcome of treat=1)', fontsize=10)
plt.title('General Train Result', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()
```
yyplot：
・DoWhy経由で学習した結果
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ce27993c-76ba-3d47-7959-648f843379e4.png)
・普通に学習した結果
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ba60e189-0450-fb54-24ab-59efe48a7bfb.png)

直接EconMLで学習させたモデルで、実測と予測の散布図yyplotを可視化。
上図の普通に予測モデルを作った場合の結果と一致する結果となり、DoWhy経由のモデルよりも精度が良い。
```python
# データ
dataset = dataset_init.copy()
# TLearnerで使用するモデル（LightGBM）
clf0 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)  # treat=0のモデル
clf1 = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)  # treat=1のモデル
est = TLearner(models=[clf0, clf1])  # EconML TLearner Object
# 学習
est.fit(dataset[target_col], dataset[tr_col], X=dataset[exp_col])

clf0 = est.models[0]  # treat=0の学習済みモデル
clf1 = est.models[1]  # treat=1の学習済みモデル

# treat=1のサンプル
df1 = dataset[dataset[tr_col]==1].reset_index(drop=True).copy()
# treat=0のサンプル
df0 = dataset[dataset[tr_col]==0].reset_index(drop=True).copy()

### 実測予測プロット ###
fig=plt.figure(figsize=(5,5))
plt.scatter(df1[target_col].to_numpy(), clf1.predict(df1[exp_col].to_numpy()))
plt.xlim(0, max(df1[target_col].to_numpy().max(), clf1.predict(df1[exp_col].to_numpy()).max()))
plt.ylim(0, max(df1[target_col].to_numpy().max(), clf1.predict(df1[exp_col].to_numpy()).max()))
plt.xlabel('actual (outcome of treat=1)', fontsize=10)
plt.ylabel('predict (outcome of treat=1)', fontsize=10)
plt.title(est.__class__.__name__+': EconML Train Result', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()
```
yyplot：
・EconMLで学習した結果
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e091140b-66c8-4c51-958c-94e760883955.png)

※上の3つのyyplotを並べるとDoWhy経由は明らかにおかしいことがわかる
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/4395438c-4cad-5f1b-c8d4-8ff739c34ee7.png)


S-Learnerも同様に確認。
T-Learnerの時と同じく直接EconMLで学習させたモデルは普通に予測モデルを作った場合の結果と一致する結果となり、DoWhy経由のモデルよりも精度が良い。
```python
# データ
dataset = dataset_init.copy()
# DAGをもとにモデル定義
model= dowhy.CausalModel(data = dataset
                         , graph=causal_graph.replace("\n", " ")
                         , treatment=tr_col
                         , outcome=target_col)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# TLearnerで使用するモデル（LightGBM）
clf = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
SLearner_att = model.estimate_effect(identified_estimand
                                     , method_name="backdoor.econml.metalearners.SLearner"
                                     #, target_units=lambda df: df[tr_col]>0  # CATE(treat=1)  # CATEの条件指定可
                                     , method_params={'init_params':{'overall_model':clf}
                                                      , 'fit_params':{}
                                                     }
                                    )

learner = SLearner_att.params['_estimator_object']  # EconML TLearner Object
clf = learner.overall_model  # 学習済みモデル

tr1_ind = dataset[dataset[tr_col]==1].index  # treat=1のindex番号
tr0_ind = dataset[dataset[tr_col]==0].index  # treat=0のindex番号

### EconML SLearnerは内部でTreat変数をOneHotしているので同様に加工 ###
# treat=1用
df1 = dataset.copy()
df1.drop(columns=[target_col, tr_col], inplace=True)
df1[tr_col+'_inv'] = 0
df1[tr_col] = 1
# treat=0用
df0 = dataset.copy()
df0.drop(columns=[target_col, tr_col], inplace=True)
df0[tr_col+'_inv'] = 1
df0[tr_col] = 0

### 実測予測プロット ###
# DoWhy経由で実施すると明らかにmodelの精度が悪い
fig=plt.figure(figsize=(5,5))
plt.scatter(dataset[dataset[tr_col]==1][target_col].to_numpy(), clf.predict(df1.iloc[tr1_ind,:].to_numpy()))
plt.xlim(0, max(dataset[dataset[tr_col]==1][target_col].to_numpy().max(), clf.predict(df1.iloc[tr1_ind,:].to_numpy()).max()))
plt.ylim(0, max(dataset[dataset[tr_col]==1][target_col].to_numpy().max(), clf.predict(df1.iloc[tr1_ind,:].to_numpy()).max()))
plt.xlabel('actual (outcome of treat=1)', fontsize=10)
plt.ylabel('predict (outcome of treat=1)', fontsize=10)
plt.title(learner.__class__.__name__+': DoWhy Train Result', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()


# 普通に予測モデルを学習させると精度はいい
clf = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)
clf.fit(dataset[exp_col+[tr_col]], dataset[target_col])

fig=plt.figure(figsize=(5,5))
plt.scatter(dataset[dataset[tr_col]==1][target_col].to_numpy()
            , clf.predict(dataset[dataset[tr_col]==1][exp_col+[tr_col]].to_numpy()))
plt.xlim(0, max(dataset[dataset[tr_col]==1][target_col].to_numpy().max(), clf.predict(dataset[dataset[tr_col]==1][exp_col+[tr_col]].to_numpy()).max()))
plt.ylim(0, max(dataset[dataset[tr_col]==1][target_col].to_numpy().max(), clf.predict(dataset[dataset[tr_col]==1][exp_col+[tr_col]].to_numpy()).max()))
plt.xlabel('actual (outcome of treat=1)', fontsize=10)
plt.ylabel('predict (outcome of treat=1)', fontsize=10)
plt.title('General Train Result', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()
```
yyplot：
・DoWhy経由で学習した結果
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d568e3c7-0805-3c36-5e4c-d86013ca0381.png)
・普通に学習した結果
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f1abd1e5-482c-df1f-07ef-a9302da3af1a.png)

```python
# データ
dataset = dataset_init.copy()
# SLearnerで使用するモデル（LightGBM）
clf = LGBMRegressor(n_estimators=1000, max_depth=10, random_state=0)  # モデル
est = SLearner(overall_model=clf)  # EconML SLearner Object
# 学習
est.fit(dataset[target_col], dataset[tr_col], X=dataset[exp_col])

clf = est.overall_model  # 学習済みモデル

tr1_ind = dataset[dataset[tr_col]==1].index  # treat=1のindex番号
tr0_ind = dataset[dataset[tr_col]==0].index  # treat=0のindex番号

### EconML SLearnerは内部でTreat変数をOneHotしているので同様に加工 ###
# treat=1用
df1 = dataset.copy()
df1.drop(columns=[target_col, tr_col], inplace=True)
df1[tr_col+'_inv'] = 0
df1[tr_col] = 1
# treat=0用
df0 = dataset.copy()
df0.drop(columns=[target_col, tr_col], inplace=True)
df0[tr_col+'_inv'] = 1
df0[tr_col] = 0

### 実測予測プロット ###
fig=plt.figure(figsize=(5,5))
plt.scatter(dataset[dataset[tr_col]==1][target_col].to_numpy(), clf.predict(df1.iloc[tr1_ind,:].to_numpy()))
plt.xlim(0, max(dataset[dataset[tr_col]==1][target_col].to_numpy().max(), clf.predict(df1.iloc[tr1_ind,:].to_numpy()).max()))
plt.ylim(0, max(dataset[dataset[tr_col]==1][target_col].to_numpy().max(), clf.predict(df1.iloc[tr1_ind,:].to_numpy()).max()))
plt.xlabel('actual (outcome of treat=1)', fontsize=10)
plt.ylabel('predict (outcome of treat=1)', fontsize=10)
plt.title(est.__class__.__name__+': EconML Train Result', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()
```
yyplot：
・EconMLで学習した結果
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/012d0ee9-9856-14d9-8699-0dc9562e2d49.png)

※上の3つのyyplotを並べるとDoWhy経由は明らかにおかしいことがわかる
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/435638cc-ad59-b125-2db3-16817918db6f.png)


このあたりの疑問点はわからないまま。しかもDoWhy経由のモデルの出力は実行する度に結果が変わるんだよなぁ。なので再度実行するとyyplotの結果が変わるという…。直接EconMLを使った場合はそんなことにはならない。ソースコードも見たが原因は突き止められなかった。
今回の介入効果の検証結果そのものはそんなに悪くなかったけど、Meta-learnerを使うときはEconMLを直接使った方が安心ではあるかも…。

# おわりに
再掲になるが、[リンク](https://www.krsk-phs.com/entry/2018/08/22/060844)より、DoWhyは以下のような思想で作られているライブラリなので、因果推論をしっかり考えながらするための良いツールだと思う。
>DoWhyではDAGを自分で書いて、読み込ませる必要があります。
あくまでDAGは自分の知識を使って書かないといけないわけで、完璧なDAGが書けるとも限りません。
すなわち、DoWhyを使えばコンピュータが勝手に因果推論を行ってくれるなんて代物ではありません。これはとても重要なポイントです。人間の頭を使う必要があります。だからこそ仮定（Assumption)なのであって、その仮定が成り立たない場合になにが起こりうるのかをあとで検討（Sensitivity Analysis)を行う必要があります。

一方、DoWhy経由でEconMLを使うには、おや？と思う点もあって困惑している。DRLearnerもバグがあるようだし。
何か使い方を間違っていたかもしれないが…。
まあ当分はMeta-learnerなどでCATEの推定をするときはEconMLを直接使うことになると思う。

以上！



