# はじめに
傾向スコアマッチングをする機会があったが、何気にpythonで簡単に実施するライブラリなど見当たらなかったので、自分で書いた。
その備忘録。

追記:
ライブラリを使用したアプローチもやってみた。↓
[DoWhyとEconMLでLalonde datasetの因果推論](https://qiita.com/chicken_data_analyst/items/61443842631c6c564ca2)

# 参考文献
[効果検証入門～正しい比較のための因果推論/計量経済学の基礎～](https://www.amazon.co.jp/dp/4297111179)
[LaLonde(1986)とその周辺の学習記録](https://moratoriamuo.hatenablog.com/entry/2020/02/10/235636)

# データ
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
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import statsmodels.api as sma
jpn_fonts=list(np.sort([ttf for ttf in fm.findSystemFonts() if 'msgothic' in ttf]))
jpn_font=jpn_fonts[0]
prop = fm.FontProperties(fname=jpn_font)
sns.set()

df_cps1 = pd.read_stata('https://users.nber.org/~rdehejia/data/cps_controls.dta')
df_nsw = pd.read_stata('https://users.nber.org/~rdehejia/data/nsw_dw.dta')
display(df_cps1.head())
display(df_nsw.head())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/512fd230-4869-8a11-3161-586ae8940bd7.png)

# 前処理
定義したdf_nswの対照群('treat'=0)をdf_cps1のデータに入れ替える。
以下のような不均衡なデータになる。
- 処置群：185人
- 対照群：15,992人

```python
# df_nswの対照群をdf_cps1に入れ替える
tr_col = 'treat' # 介入変数
all_col = ['age','education','black','hispanic','married','nodegree','re74','re75','re78']
exp_col = ['age','education','black','hispanic','married','nodegree','re74','re75'] # 共変量
df_nswcps1 = pd.concat([df_nsw[df_nsw[tr_col] == 1], df_cps1], ignore_index=False).reset_index(drop=True)
display(df_nswcps1)
display(df_nswcps1.treat.value_counts().reset_index())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/056800a7-acd1-7bb0-e2f1-a23338946cb5.png)

# 傾向スコアの計算
ロジスティック回帰で傾向スコアを求める。
```python
# 傾向スコアのためのモデル作成関数
def model_lr(X_train_std, y_train):
    model = LogisticRegression(max_iter=10000, C=500) # ロジスティック回帰
    model.fit(X_train_std, y_train)

    pred = model.predict(X_train_std)
    pred_proba = model.predict_proba(X_train_std)
    return model, pred, pred_proba[:,1]
```

共変量を標準化して学習。
```python
X_train = df_nswcps1[exp_col].copy() # 共変量
y_train = df_nswcps1[tr_col].copy() # 介入変数
sc = sklearn.preprocessing.StandardScaler() # 標準化
X_train_std = sc.fit_transform(X_train) # 標準化
# ロジスティック回帰
model, pred, pred_proba = model_lr(X_train_std, y_train)
```

傾向スコアでマッチングしていくためのdf作成。
```python
result1 = df_nswcps1.copy()
result1['user_id'] = range(len(result1)) #user_id付与
result1['Zscore'] = pred_proba # 傾向スコア
# user_id、介入変数、傾向スコア、共変量のデータフレーム
result1 = result1[['user_id', tr_col, 'Zscore']+exp_col]
```

# 傾向スコアマッチング
傾向スコアマッチングをする関数。
numpyのargminで処置群の傾向スコアと最も近い傾向スコアを持つ対照群を取得する。
マッチング時に許容する差(キャリパー)は、$(データ全体の傾向スコアの標準偏差)×0.2$
に設定している。
やり方に迷ってしまったのが、マッチングした後の対照群のサンプルについて、重複してマッチングしないようにすること。
群からサンプルを消してしまうと次からインデックス番号がずれるので、消さずに‐9999という絶対にマッチングしない数値に置換してしまうという戦法にした。
作った後に、「ループ中に毎回knnでfitしてしまえばよかったような…」と気づいたが無視した。
```python
# 傾向スコアマッチング関数
def matching(result1, z_col='Zscore', tr_col='flg', match_id='user_id'):
    '''
    result1: 介入変数、傾向スコア、共変量のデータフレーム
    '''
    # キャリパー設定
    caliper = result1[z_col].std()*0.2

    # 処置群抽出
    Tr = result1[result1[tr_col]>0][[match_id, z_col]].copy().reset_index(drop=True)
    Tr.index = Tr[match_id]
    # 対照群抽出
    Ctl = result1[result1[tr_col]==0][[match_id, z_col]].copy().reset_index(drop=True)
    Ctl.index = Ctl[match_id]
    
    Ctl_vals = np.copy(Ctl[z_col].to_numpy()) # 配列化
    Ctl_ids = np.copy(Ctl.index.to_numpy()) # 配列化
    Tr_vals_dict = Tr[z_col].sort_values(ascending=False).to_dict() # 処置群の傾向スコアで降順にして辞書化

    # マッチングを実施していく
    match_results = [] # 結果を入れる箱
    for i, (cid, tr_val) in tqdm(enumerate(Tr_vals_dict.items())):
        # cid:user_id, tr_val:傾向スコア
        # argminで処置群と最も傾向スコアの差が小さい対照群のインデックス番号取得
        nearId = np.abs(Ctl_vals - tr_val).argmin()
        # 傾向スコアの差がキャリパーより大きかった場合、マッチングしない
        if np.abs(tr_val-Ctl_vals[nearId])>caliper:
            continue
        # 傾向スコアの差がキャリパーより小さかった場合、マッチングした者同士をリストに格納
        match_results.append([cid, Ctl_ids[nearId], tr_val, Ctl_vals[nearId]])
        # 次のマッチングで重複してマッチングしないように対照群のスコアを-9999に置き換える
        np.put(Ctl_vals, [nearId], -9999)

    # リストをdf化
    match_results_df = pd.DataFrame(match_results, columns=['Tr', 'Ctl', 'Tr_zscore', 'Ctl_zscore'])
    match_results_df['diff'] = np.abs(match_results_df['Tr_zscore']-match_results_df['Ctl_zscore'])
    print(match_results_df['diff'].max())
    return match_results_df

match_results_df = matching(result1, z_col='Zscore', tr_col='treat', match_id='user_id')
display(match_results_df)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/50539848-c1c1-9d2f-b9f1-3505444222d5.png)

Trは処置群のuser_id、Ctlは対照群のuser_id、またそれぞれの傾向スコアと、その差分が入ったデータフレームである。
183個のサンプルについてマッチングできた。
もともと処置群は185人だったので2人はキャリパーによってマッチングしなかった。
対照群は15,992人からマッチングした183人に絞られている。

# マッチング前後比較
## 傾向スコアの分布
対照群と処置群の傾向スコアの分布を確認、
マッチングがうまくいっていたら、同じような分布になっているはず。
```python
# 傾向スコアのヒストグラム
def zscore_hist(df01, df02, lim=None, vmax=None):
    fig=plt.figure(figsize=(6,6))
    ax = plt.subplot(1,1,1)
    ax2 = ax.twinx()
    sns.histplot(df01, binwidth=0.05, binrange=(0,1), ax=ax, kde=False, label='0', color ='b', alpha=1.)
    sns.histplot(df02, binwidth=0.05, binrange=(0,1), ax=ax2, kde=False, label='1', color ='r', alpha=0.5)
    ax2.grid(False)
    if lim=='zoom':
        ax.set_ylim(0, len(df02)*1)
    elif lim=='same':
        ax.set_ylim(0, len(df02))
        ax2.set_ylim(0, len(df02))
    elif lim=='custom':
        ax.set_ylim(0, vmax)
        ax2.set_ylim(0, vmax)
    else:
        pass
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)
    plt.show()
```

マッチング前。
```python
# 傾向スコアヒストグラム マッチング前
df01 = result1[result1[tr_col]==0]['Zscore']
df02 = result1[result1[tr_col]>0]['Zscore']
zscore_hist(df01, df02, lim=False)
zscore_hist(df01, df02, lim=True) # 縦軸拡大
```
左縦軸：対照群
右縦軸：処置群
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f6aa5140-816e-cb90-2e51-a24dd1dda53f.png)
左縦軸拡大
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/b22456e4-05a9-a7cd-0855-77cb9985611a.png)

マッチング後。
```python
# 傾向スコアヒストグラム マッチング後
df01 = result1[result1['user_id'].isin(match_results_df['Ctl'].unique())]['Zscore']
df02 = result1[result1['user_id'].isin(match_results_df['Tr'].unique())]['Zscore']
zscore_hist(df01, df02, lim='custom', vmax=60)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/b71a7b5c-f9df-310b-237a-4b77c2f654b1.png)

マッチング前よりも分布は揃った、

## 共変量のヒストグラム
共変量の分布をマッチング前後で確認。
マッチングがうまくいっていたら、同じような分布になっているはず。
```python
# 共変量のヒストグラム
def histhist(result1, Ctls, Trs, exp_cols, wid_ratio=20, match=False):
    dim = len(exp_cols)
    fig=plt.figure(figsize=(21,14))
    plt.rcParams['font.family'] = prop.get_name()
    for i, col in tqdm(enumerate(exp_cols)):
        vmin=result1[col].min()
        vmax=result1[col].max()
        VAL = Ctls[col].copy()
        VAL2 = Trs[col].copy()

        wid = round(vmax/wid_ratio)
        if wid==0 or round(wid)==0:
            wid=0.1

        ax = plt.subplot(round(np.ceil(dim/np.sqrt(dim))), round(np.ceil(dim/np.sqrt(dim))), i+1)
        ax2 = ax.twinx()
        sns.histplot(VAL.to_numpy()
                     , binwidth=wid
                     , binrange=(0,vmax)
                     , ax=ax, kde=False, label='0', color ='b', alpha=1.)
        sns.histplot(VAL2.to_numpy()
                     , binwidth=wid
                     , binrange=(0,vmax)
                     , ax=ax2, kde=False, label='1', color ='r', alpha=0.5)
        ax2.grid(False)
        if match:
            ax.set_ylim(0, len(VAL2))
            ax2.set_ylim(0, len(VAL2))
        ax.set_title(col+', binwidth:'+str(np.round(wid, 3)))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()
```

マッチング前。
```python
# マッチング前
Ctls = df_nswcps1[df_nswcps1[tr_col]==0].copy() # 対照群
Trs = df_nswcps1[df_nswcps1[tr_col]>0].copy() # 処理群
histhist(df_nswcps1, Ctls, Trs, exp_col) # 共変量ヒストグラム マッチング前
```
左縦軸：対照群
右縦軸：処置群
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/64bcdf4f-b390-092c-f995-95bd5f1d46ba.png)

マッチング後。
```python
# マッチング後
Ctls = result1[result1['user_id'].isin(match_results_df['Ctl'].unique())].copy() # 対照群
Trs = result1[result1['user_id'].isin(match_results_df['Tr'].unique())].copy() # 処理群
histhist(result1, Ctls, Trs, exp_col, match=True) # 共変量ヒストグラム マッチング後
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d275c879-7f5b-25c5-ee81-d5f6b6ca81ab.png)

マッチング前の共変量は明らかに処置群と対照群で分布が大きく違っていたが、マッチング後には処置群と対照群の分布が近づいたように見える。

## 標準化平均差(ASAM)
標準化平均差(ASAM)を見ることで処置群と対照群で各変数の平均にどの程度の差があるのか確認できる。
マッチング前後のASAMを見ることで、マッチングにより処置群と対照群の共変量の差がどの程度無くなったか確認できる。
多くの場合で、ASAMが0.1以下の時十分に共変量のバランスが取れていると考えられている。

```python
# 標準化平均差プロット
def love_plot(result1, match_results_df, exp_cols
              , tr_col='flg', match_id='user_id', visual=True):
    # 処置群と対照群の共変量の平均の差
    asamCtlMean = result1[result1[tr_col]==0][exp_cols].mean()-result1[result1[tr_col]>0][exp_cols].mean()
    # 共変量の標準偏差
    asamCtlStd =result1[exp_cols].std() 
    # 標準化平均差
    asamCtl = asamCtlMean.abs()/asamCtlStd
    
    # マッチング後の処置群と対照群の共変量の平均の差
    asamTrMean = result1[result1[match_id].isin(match_results_df['Ctl'].unique())][exp_cols].mean()-result1[result1[match_id].isin(match_results_df['Tr'].unique())][exp_cols].mean()
    # マッチング後の共変量の標準偏差
    asamTrStd =result1[exp_cols].std() 
    # マッチング後の標準化平均差
    asamTr = asamTrMean.abs()/asamTrStd
    if visual:
        fig=plt.figure(figsize=(6,6))
        plt.rcParams['font.family'] = prop.get_name()
        plt.axvline(0.1, ls='--', c='gray', alpha=0.8)
        plt.axvline(0.25, ls='--', c='gray', alpha=0.8)
        plt.axvline(0.4, ls='--', c='gray', alpha=0.8)
        plt.plot(asamCtl.to_numpy(), asamCtl.index, marker='x', ls='', label='before matching')
        plt.plot(asamTr.to_numpy(), asamTr.index, marker='o', ls='', label='after matching', alpha=0.6)
        plt.legend()
        plt.title('ASAM')
        plt.tight_layout()
        plt.show()
    return asamCtl, asamTr, asamCtlMean, asamTrMean

asamCtl, asamTr, asamCtlMean, asamTrMean = love_plot(result1, match_results_df, exp_col
                                                     , tr_col=tr_col, match_id='user_id', visual=True)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/877499e5-aaa5-fe5c-d0b6-0019233a5b31.png)

マッチング前に比べて、マッチング後の方がASAMが小さくなっており、多くの変数で0.1を下回り、すべての変数が0.25を下回っている。

## 介入効果の確認
NSWのRCTによると、're78'への介入効果は$1,676であり統計的にも有意。
以下のように回帰分析で確認できる。
```python
# 回帰分析NSW
# statsmodelでは、切片を必要とする線形回帰のモデル式の場合、全要素が1.0の列を説明変数の先頭に追加する必要がある
X = sma.add_constant(df_nsw[exp_col+[tr_col]]) # 全要素が1.0の列を先頭に追加
est = sma.OLS(df_nsw['re78'], X)
est_trained = est.fit()
print(est_trained.summary())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/dca8da28-c4ef-4ba9-af7e-a8e72ede71e5.png)


今回分析したNSWの一部をCPSに置き換えたデータについて、単純に回帰分析すると、セレクションバイアスにより効果は$699となり、適切な効果測定ができていない。
回帰分析の係数も有意ではない。
```python
# 回帰分析NSW+CPS(マッチング前)
X = sma.add_constant(df_nswcps1[exp_col+[tr_col]])
est = sma.OLS(df_nswcps1['re78'], X)
est_trained = est.fit()
print(est_trained.summary())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/b8fc7f42-7fb7-1851-41a4-bf97cf870baf.png)

マッチング後の介入効果を見ると、効果は$1,704とRCTの結果に近づき、係数も有意となっている。
```python
# 回帰分析NSW+CPS(マッチング後)
df_nswcps1_match = df_nswcps1.copy()
df_nswcps1_match['user_id'] = range(len(df_nswcps1_match))
df_nswcps1_match = df_nswcps1_match[(df_nswcps1_match['user_id'].isin(match_results_df['Ctl'].unique()))\
                                    |(df_nswcps1_match['user_id'].isin(match_results_df['Tr'].unique()))].copy()
X = sma.add_constant(df_nswcps1_match[exp_col+[tr_col]])
est = sma.OLS(df_nswcps1_match['re78'], X)
est_trained = est.fit()
print(est_trained.summary())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/93e90257-e75f-cc33-9de8-0d072dfaea63.png)

ということで傾向スコアマッチングはうまくいったようだ。

# おわりに
pythonで傾向スコアマッチングを実施した。
実は知らないだけでサクッとやってくれるライブラリとかあるんだろうか。
あったらそれ使いたいなーと思った。

以上！
