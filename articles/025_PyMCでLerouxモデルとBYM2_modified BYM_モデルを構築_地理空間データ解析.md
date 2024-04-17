# はじめに
最近、空間統計のお勉強をしているので、また空間統計モデルのお話をする。  
以前[「衛星データでつくばみらい市の土壌分類をしてみた（地理空間データ解析）」](https://qiita.com/chicken_data_analyst/items/886d35561a4f23653dc4)という記事で、条件付き自己回帰モデル（Conditional Auto-Regressive model;CAR model）の1つ、ICAR（Intrinsic CAR）モデルについて書いたが、CARモデルには他にも種類があってICARモデルの欠点が修正されていたりする。今回はその中でもLerouxモデルとBYM2モデルを構築して、またつくばみらい市の土壌分類を実施。  
土壌分類は以下のような感じのデータ。（図：NDVIと土壌クラスの画像）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/77ce0d5e-a4c8-392e-67ef-37a24e8cbd44.png)

過去の空間統計に関する記事は以下。 

- [衛星データでつくばみらい市の変化を見てみた（地理空間データ解析）](https://qiita.com/chicken_data_analyst/items/ed3a6002e82d4ea63556)
- [衛星データでつくばみらい市の土壌分類をしてみた（地理空間データ解析）](https://qiita.com/chicken_data_analyst/items/886d35561a4f23653dc4)

やることを分けると以下の順で7つある。
1. 衛星データダウンロード
2. 小地域の境界データダウンロード
3. 衛星データの前処理
4. 土壌分類データダウンロード
5. バンド演算
6. Lerouxモデル構築
7. BYM2モデル構築

1~5については上記2つの過去の空間統計に関する記事と同様なので、説明は割愛する。（コードだけ載せておく。）

# 参考
## 前回も参考にしたもの
- [[書籍]Pythonで学ぶ衛星データ解析基礎 環境変化を定量的に把握しよう](https://gihyo.jp/book/2022/978-4-297-13232-3)
- [[書籍]Rではじめる地理空間データの統計解析入門](https://www.kspub.co.jp/book/detail/5273036.html)
- [[Github]Pythonで学ぶ衛星データ解析基礎 環境変化を定量的に把握しよう](https://github.com/tamanome/satelliteBook/tree/main/notebooks)
- [Rasterio公式](https://rasterio.readthedocs.io/en/stable/)
- [GeoPandas公式](https://geopandas.org/en/stable/)
- [rasterstats公式](https://pythonhosted.org/rasterstats/)
- [EarthPy公式](https://earthpy.readthedocs.io/en/latest/)
- [Shapely公式](https://shapely.readthedocs.io/en/stable/manual.html)
- [PySAL公式](https://pysal.org/)
- [libpysal公式](https://pysal.org/libpysal/)
- [Raster Resampling for Discrete and Continuous Data](https://gisgeography.com/raster-resampling/)
- [Copernicus Browser（Sentinel衛星のデータをダウンロードできる）](https://browser.dataspace.copernicus.eu/)
- [GISデータ一覧　行政区域データ](https://note.sngklab.jp/?p=301)
- [境界データ（市町村）：国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2024.html)
- [境界データ（小地域（町丁・字等別））：e-Stat](https://www.e-stat.go.jp/gis/statmap-search?page=1&type=2&aggregateUnitForBoundary=A&toukeiCode=00200521&toukeiYear=2020&serveyId=A002005212020&coordsys=1&format=shape&datum=2011)
- [EPSGコード一覧表/日本でよく利用される空間座標系（座標参照系）](https://lemulus.me/column/epsg-list-gis)
- [課題に応じて変幻自在？ 衛星データをブレンドして見えるモノ・コト #マンガでわかる衛星データ](https://sorabatake.jp/5192/)
- [衛星スペクトル指標を用いた都市化の画像解析 ](https://www.cit.nihon-u.ac.jp/kouendata/No.39/3_doboku/3-024.pdf)
- [森林分野における衛星データ利用事例](https://tellusxdp.github.io/start-python-with-tellus/tellus_forest.html)
- [漁業での衛星データ利用事例](https://tellusxdp.github.io/start-python-with-tellus/tellus_fishery.html)
- [Sentinel-2 Imagery: NDVI Raw](https://www.arcgis.com/home/item.html?id=1e5fe250cdb8444c9d8b16bb14bd1140)
- [Sentinel-2 Imagery: Normalized Difference Built-Up Index (NDBI)](https://www.arcgis.com/home/item.html?id=3cf4e98f035e47279091dc74d43392a5)
- [esrij ジャパン GIS 基礎解説 リサンプリング](https://www.esrij.com/gis-guide/imagery/resampling/)
- [つくばみらい市のホームページ](https://www.city.tsukubamirai.lg.jp/page/page002206.html)
- [国土数値情報ダウンロードサイト 土地利用細分メッシュデータ](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L03-b-2021.html)
- [ICARモデルのMCMCをPythonで実行する](https://qiita.com/hbk24/items/32784ff02de01ec0fb95)
- [NumPyro：再パラメータ化](https://zenn.dev/yoshida0312/articles/e3709c3a77c40a)
- [実験！岩波データサイエンス1のベイズモデリングをPyMC Ver.5で⑨空間自己回帰モデル：1次元の個体数カウントデータ](https://note.com/e_dao/n/nb084e0dd057f)
- [実験！岩波データサイエンス1のベイズモデリングをPyMC Ver.5で⑩空間自己回帰モデル：2次元の株数カウントデータ](https://note.com/e_dao/n/nc6ae5321d34e?magazine_key=ma0ad746110eb)
- [Bayesian Analysis with PyMC 勉強ノート 5 分類モデル](https://zenn.dev/inaturam/articles/9d2844296a89cc)
- [衛星データでつくばみらい市の変化を見てみた（地理空間データ解析）](https://qiita.com/chicken_data_analyst/items/ed3a6002e82d4ea63556)
- [階層ベイズで個性を捉える(PyMC ver.5.7.2)](https://qiita.com/chicken_data_analyst/items/097fe82b6a8804b59924)
- [[書籍]RとStanではじめる　ベイズ統計モデリングによるデータ分析入門](https://www.kspub.co.jp/book/detail/5165362.html)
- [PyMC公式](https://www.pymc.io/welcome.html)
- [[PyMC Discourse Forum]How to perform weighted inference](https://discourse.pymc.io/t/how-to-perform-weighted-inference/1825/2)
- [[PyMC Discourse Forum]How to run logistic regression with weighted samples](https://discourse.pymc.io/t/how-to-run-logistic-regression-with-weighted-samples/5689)
- [[PyMC Discourse Forum]Pm.sample_posterior_predictive() not working with weights](https://discourse.pymc.io/t/pm-sample-posterior-predictive-not-working-with-weights/5698)
- [[PyMC Discourse Forum]UserWarning: The effect of Potentials on other parameters is ignored during prior predictive sampling. This is likely to lead to invalid or biased predictive samples](https://discourse.pymc.io/t/userwarning-the-effect-of-potentials-on-other-parameters-is-ignored-during-prior-predictive-sampling-this-is-likely-to-lead-to-invalid-or-biased-predictive-samples/10934)
- [[PDF file]条件付自己回帰モデルによる空間自己相関を考慮した生物の分布データ解析](https://www.jstage.jst.go.jp/article/seitai/59/2/59_KJ00005653149/_pdf)
- [[書籍]基礎からわかるリモートセンシング](https://www.rikohtosho.co.jp/book/721/)
- [Sentinel-2A / 2B / 2C / 2D 概要](https://www.restec.or.jp/satellite/sentinel-2-a-2-b.html)
- [esri 指数ギャラリー](https://pro.arcgis.com/ja/pro-app/3.1/help/data/imagery/indices-gallery.htm)
- [Index Formula List](https://rindcalc.readthedocs.io/en/latest/Index%20Formula%20List.html)
- [[PDF file]時系列LANDSATデータによる足尾荒廃山地における植生回復モニタリング](https://www.jstage.jst.go.jp/article/jsprs/60/4/60_200/_pdf)
- [(研究成果) 水田の代かき時期を衛星データで広域把握 用語の解説：MNDWI](https://www.naro.go.jp/publicity_report/press/laboratory/nire/135613.html)
- [[PDF file]尾瀬地域における衛星リモートセンシングによる植生モニタリング手法の検討](https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/84980/1/22_p343-352_LT80.pdf)
- [[PDF file]植物群落内放射伝達モデルを用いたマルチスペクトルカメラの違いによる水田観測結果への影響評価](https://www.jstage.jst.go.jp/article/air/31/3/31_65/_pdf/-char/ja)
- [[PDF file]Impervious Surface Extraction by Linear Spectral Mixture Analysis with Post-Processing Model](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9139358)
- [[PDF file]Developing soil indices based on brightness, darkness, and greenness to improve land surface mapping accuracy](https://www.tandfonline.com/doi/pdf/10.1080/15481603.2017.1328758)

## 今回から参考にしたもの
- [[Github]ICAR class](https://www.pymc.io/projects/docs/en/v5.13.0/_modules/pymc/distributions/multivariate.html#ICAR)
- [Spatial Models in Stan: Intrinsic Auto-Regressive Models for Areal Data](https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
- [Exact sparse CAR models in Stan](https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html)
- [An intuitive Bayesian spatial model for disease mapping that accounts for scaling](https://arxiv.org/pdf/1601.01180.pdf)
- [How to accelerate spatial models on STAN](https://discourse.mc-stan.org/t/how-to-accelerate-spatial-models-on-stan/17961/1)
- [A Bayesian hierarchical model for disease mapping that accounts for scaling and heavy-tailed latent effects](https://arxiv.org/pdf/2109.10330.pdf)
- [Investigation of Bayesian spatial models](https://cancerqld.blob.core.windows.net/content/docs/Investigation-of-Bayesian-spatial-models.pdf)
- [The Besag-York-Mollie Model for Spatial Data](https://www.pymc.io/projects/examples/en/latest/spatial/nyc_bym.html)

# 環境
使用環境はGoogle Colaboratory。  
以下のパッケージをpipでインストールする。（使っていないやつもあるけど…。）  
```python
!pip install geopandas
!pip install earthpy
!pip install rasterio
!pip install sentinelsat
!pip install cartopy
!pip install fiona
!pip install shapely
!pip install pyproj
!pip install pygeos
!pip install rtree
!pip install rioxarray
!pip install pystac-client sat-search
!pip install rich
!pip install rasterstats
!pip install pysal
!pip install libpysal
!pip install geoplot
!pip install optuna
!pip install optuna-integration
!pip install japanize-matplotlib
```

Google Driveをマウントしておく。
```python
from google.colab import drive
drive.mount('/content/drive')
```

使用パッケージimport
```python
#必要ライブラリのインポート
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ptick
import matplotlib.font_manager as fm
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import japanize_matplotlib

import seaborn as sns
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sklearn
import sklearn.cluster
import sklearn.metrics
from sklearn import preprocessing
import rich.table
import xarray as xr
import rioxarray as rxr

import geopandas as gpd
import geoplot as gplt
import cartopy, fiona, shapely, pyproj, rtree, pygeos
import cv2
import rasterio as rio
from rasterio import plot
from rasterio.plot import show, plotting_extent
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from shapely.geometry import MultiPolygon, Polygon, shape
import rasterstats
import earthpy.spatial as es
import earthpy.plot as ep
import libpysal

import os
import zipfile
import glob
import shutil

import urllib
from satsearch import Search
from pystac_client import Client
from PIL import Image
import requests
import io
from pystac_client import Client
import warnings
# from tqdm import tqdm
from tqdm.notebook import tqdm

import pymc as pm
import jax
import arviz as az
import pytensor.tensor as pt
import pytensor
import gc
import optuna.integration.lightgbm as opt_lgb

# CPU Multi
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
print(jax.default_backend())
print(jax.devices("cpu"))

# 日本語フォント読み込み
japanize_matplotlib.japanize()
# jpn_fonts=list(np.sort([ttf for ttf in fm.findSystemFonts() if 'ipaexg' in ttf or 'msgothic' in ttf or 'japan' in ttf or 'ipafont' in ttf]))
# jpn_font=jpn_fonts[0]
jpn_font = japanize_matplotlib.get_font_ttf_path()
prop = fm.FontProperties(fname=jpn_font)
print(jpn_font)
plt.rcParams['font.family'] = prop.get_name() #全体のフォントを設定
plt.rcParams['figure.dpi'] = 250

pd.options.display.float_format = '{:.5f}'.format
sns.set()
```

# 衛星データダウンロード～バンド演算まで

<details><summary>衛星データダウンロード～バンド演算まで(折り畳み)</summary>

```python
# データをダウンロードするかどうか
DOWNLOAD = False

# 保存するディレクトリの作成
dst_dir = '/content/drive/MyDrive/satelite/s2Bands'  # Google Colabでは'/content~~'が正
os.makedirs(dst_dir, exist_ok=True)

# 座標の基準とするラスターデータ読み込み
raster_crs = rio.open(os.path.join(dst_dir,'S2B_54SVE_20180428_0_L2A_B11.tif'))
raster_profile = raster_crs.profile

# 小地域区分のベクターデータ（from e-Stat）を読み込みcrsをラスターデータに合わせる
shape_path = "/content/drive/MyDrive/satelite/ibrakiPolygon/"  # Google Colabでは'/content~~'が正
os.makedirs(shape_path, exist_ok=True)
part_in_shape = gpd.read_file(os.path.join(shape_path, "r2ka08235.shp"), encoding="shift-jis")[['PREF_NAME', 'CITY_NAME', 'S_NAME', 'AREA', 'PERIMETER', 'JINKO', 'SETAI', 'geometry']]
re_shape_tsukuba_mirai_2RasterCrs = part_in_shape.to_crs(raster_profile["crs"])  #crs合わせ
print(re_shape_tsukuba_mirai_2RasterCrs.shape)
# 同じ小地域がさらに細かく分かれている場合があるので、小地域単位でグループ化しておく
re_shape_tsukuba_mirai_2RasterCrs = re_shape_tsukuba_mirai_2RasterCrs.dissolve(['PREF_NAME', 'CITY_NAME', 'S_NAME'], aggfunc='sum', as_index=False)
print(re_shape_tsukuba_mirai_2RasterCrs.shape)  # 特定の小地域が統合されレコード数が減る
display(re_shape_tsukuba_mirai_2RasterCrs)

# 取得範囲を指定するための関数を定義
def selSquare(lon, lat, delta_lon, delta_lat):
    c1 = [lon + delta_lon, lat + delta_lat]
    c2 = [lon + delta_lon, lat - delta_lat]
    c3 = [lon - delta_lon, lat - delta_lat]
    c4 = [lon - delta_lon, lat + delta_lat]
    geometry = {"type": "Polygon", "coordinates": [[ c1, c2, c3, c4, c1 ]]}
    return geometry

# 茨城県周辺の緯度経度をbbox内へ
geometry = selSquare(140.0363, 35.9632, 0.06, 0.02)
timeRange = '2018-01-01/2023-12-31' # 取得時間範囲を指定

# STACサーバに接続し、取得範囲・時期やクエリを与えて取得するデータを絞る
# sentinel:valid_cloud_coverを用いて、雲量の予測をより確からしいもののみに限定している
if DOWNLOAD:
  api_url = 'https://earth-search.aws.element84.com/v0'
  collection = "sentinel-s2-l2a-cogs"  # Sentinel-2, Level 2A (BOA)
  s2STAC = Client.open(api_url, headers=[])
  s2STAC.add_conforms_to("ITEM_SEARCH")

  s2Search = s2STAC.search (
      intersects = geometry,
      datetime = timeRange,
      query = {"eo:cloud_cover": {"lt": 11}, "sentinel:valid_cloud_cover": {"eq": True}},
      collections = collection)

  s2_items = [i.to_dict() for i in s2Search.get_items()]
  print(f"{len(s2_items)} のシーンを取得")

# product_idやそのgeometryの情報がまとまったdf作成
if DOWNLOAD:
  items = s2Search.get_all_items()
  df = gpd.GeoDataFrame.from_features(items.to_dict(), crs="epsg:32654")
  dfSorted = df.sort_values('eo:cloud_cover').reset_index(drop=True)
  # epsgの種類
  print('epsg', dfSorted['proj:epsg'].unique())
  display(dfSorted.head(3))
  # 雲量10以下の日時
  print(np.sort(dfSorted[dfSorted['eo:cloud_cover']<=10]['datetime'].unique()))

# 2018‐2023各年の同じ時期のproduct_id一覧取得
if DOWNLOAD:
  df_selected = dfSorted[dfSorted['datetime'].isin(['2018-04-28T01:36:01Z', '2019-05-08T01:37:21Z', '2020-05-02T01:37:21Z', '2021-04-22T01:37:12Z', '2022-04-12T01:37:20Z', '2023-04-27T01:37:18Z'])].copy().sort_values('datetime').iloc[[0,2,3,4,5,6],:]
  display(df_selected['sentinel:product_id'].to_list())

# 各productのデータURLやtifファイル名の一覧取得
if DOWNLOAD:
  selected_item = [x.assets for x in items if x.properties['sentinel:product_id'] in (df_selected['sentinel:product_id'].to_list())]
  selected_item = sorted(selected_item, key=lambda x:x['thumbnail'].href)

# thumbnailで撮影領域確認
if DOWNLOAD:
  plt.rcParams['font.family'] = prop.get_name() #全体のフォントを設定
  fig = plt.figure(figsize=(7,3))
  for ix, sitm in enumerate(selected_item):
    thumbImg = Image.open(io.BytesIO(requests.get(sitm['thumbnail'].href).content))
    ax = plt.subplot(2,3,ix+1)
    ax.imshow(thumbImg)
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    ax.set_title('撮影日時 : '+'-'.join(sitm['thumbnail'].href.split('/')[-5:-2]), fontsize=4)
    ax.grid(False)
  plt.tight_layout()
  plt.show()

# Sentinel-2のバンド情報を表で示す
if DOWNLOAD:
  table = rich.table.Table("Asset Key", "Description")
  for asset_key, asset in selected_item[0].items():
    table.add_row(asset_key, asset.title)

  display(table)

# URLからファイルをダウンロードする関数を定義
# 引用：https://note.nkmk.me/python-download-web-images/
def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file, open(dst_path, 'wb') as local_file:
            local_file.write(web_file.read())
    except urllib.error.URLError as e:
        print(e)

def download_file_to_dir(url, dst_dir):
    download_file(url, os.path.join(dst_dir, url.split('/')[-2]+'_'+os.path.basename(url)))


# tifファイルをダウンロード(時間かかる)
if DOWNLOAD:
  # 取得するバンドの選択
  bandLists = ['B12','B11','B08','B04','B03','B02'] # SWIR2, SWIR1, NIR, RED, GREEN, BLUE

  # 画像のURL取得
  file_url = []
  for sitm in selected_item:
    [file_url.append(sitm[band].href) for band in bandLists if file_url.append(sitm[band].href) is not None]

  # 画像のダウンロード
  [download_file_to_dir(link, dst_dir) for link in file_url if download_file_to_dir(link, dst_dir) is not None]

# ダウンロードファイルリスト（撮影日時順）
display(sorted(os.listdir(dst_dir), key=lambda x:(x.split('_54SVE_')[-1])))

# 試しにバンド11のデータを1つ可視化
src = rio.open(os.path.join(dst_dir,'S2B_54SVE_20180428_0_L2A_B11.tif'))
fig = plt.figure(figsize=(2, 2))
ax=plt.subplot(1,1,1)
retted = show(src.read(), transform=src.transform, cmap='RdYlGn', ax=ax, vmax=np.quantile(src.read(1), q=0.99))
img = retted.get_images()[0]
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=4)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(False)
plt.show()

# 試しにバンド12のデータを1つ可視化
src = rio.open(os.path.join(dst_dir,'S2B_54SVE_20180428_0_L2A_B12.tif'))
fig = plt.figure(figsize=(2, 2))
ax=plt.subplot(1,1,1)
retted = show(src.read(), transform=src.transform, cmap='RdYlGn', ax=ax, vmax=np.quantile(src.read(1), q=0.99))
img = retted.get_images()[0]
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params(labelsize=4)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(False)
plt.show()

# ラスターデータリスト読み込み
getList = sorted(list(glob.glob(dst_dir+'/S2*')), key=lambda x:(x.split('_54SVE_')[-1]))
display(getList)

# ラスターデータをベクターデータの範囲にcropして保存
s2Output = '/content/drive/MyDrive/satelite/s2Output'  # Google Colabでは'/content~~'が正
os.makedirs(s2Output, exist_ok=True) # outputデータ保存ディレクトリ
if DOWNLOAD:
  band_paths_list = es.crop_all(list(getList), s2Output, re_shape_tsukuba_mirai_2RasterCrs, overwrite=True)
# cropしたラスターデータリスト
band_paths_list = sorted(list(glob.glob(s2Output+'/S2*')), key=lambda x:(x.split('_54SVE_')[-1]))  # 撮影日時順にソート
display(band_paths_list)

# cropしたラスターデータ（つくばみらい市）をベクターデータと共に試しに可視化
src = rio.open(band_paths_list[3])  # Band 8
fig = plt.figure(figsize=(2, 2))
ax=plt.subplot(1,1,1)
retted = show(src.read(), transform=src.transform, cmap='coolwarm', ax=ax, vmin=0, vmax=3500)
im = retted.get_images()[0]
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", '5%')
cbar = fig.colorbar(im, ax=ax, cax=cax, shrink=0.6, extend='both')
cbar.ax.tick_params(labelsize=4)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=1, ax=ax, linewidth=0.2)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_title('_'.join(os.path.basename(src.name).split('_')[-5:-1]), fontsize=4)
ax.grid(False)
plt.tight_layout()
plt.show()

# 各バンドのtifファイルのリスト
B04s = sorted(list(glob.glob(os.path.join(s2Output,'S2*B04_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B08s = sorted(list(glob.glob(os.path.join(s2Output,'S2*B08_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B11s = sorted(list(glob.glob(os.path.join(s2Output,'S2*B11_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B12s = sorted(list(glob.glob(os.path.join(s2Output,'S2*B12_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))

B03s = sorted(list(glob.glob(os.path.join(s2Output,'S2*B03_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B02s = sorted(list(glob.glob(os.path.join(s2Output,'S2*B02_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
print(B04s)
print(B08s)
print(B11s)
print(B12s)
print(B03s)
print(B02s)

if DOWNLOAD:
    # バンドごとに分解能が違う場合があるので、リサンプリングしてグリッドを合わせる
    # 10m分解能のバンドを20m分解能のバンド11のグリッドに合わせる
    for num, (b04, b08, b11, b12, b03, b02) in enumerate(zip(B04s, B08s, B11s, B12s, B03s, B02s)):
      print('#####', os.path.basename(b08).split('_')[2], '#####')
      riod04 = rio.open(b04)
      riod08 = rio.open(b08)
      riod11 = rio.open(b11)
      riod12 = rio.open(b12)
      riod03 = rio.open(b03)
      riod02 = rio.open(b02)
      bounds = riod04.bounds
      # print(riod08.read().shape)
      # print(riod11.read().shape)
      # ラスターデータをリサンプリングしてバンド11に合わせる
      riod08_resampling = riod08.read(out_shape=(riod08.count,int(riod11.height),int(riod11.width)),
                                      resampling=Resampling.cubic)
      riod04_resampling = riod04.read(out_shape=(riod04.count,int(riod11.height),int(riod11.width)),
                                      resampling=Resampling.cubic)
      riod03_resampling = riod03.read(out_shape=(riod03.count,int(riod11.height),int(riod11.width)),
                                      resampling=Resampling.cubic)
      riod02_resampling = riod02.read(out_shape=(riod02.count,int(riod11.height),int(riod11.width)),
                                      resampling=Resampling.cubic)
    
      print('B11', riod11.read().shape, riod11.read().shape, sep='-->')
      print('B12', riod12.read().shape, riod12.read().shape, sep='-->')
      print('B04', riod04.read().shape, riod04_resampling.shape, sep='-->')
      print('B08', riod08.read().shape, riod08_resampling.shape, sep='-->')
      print('B03', riod03.read().shape, riod03_resampling.shape, sep='-->')
      print('B02', riod02.read().shape, riod02_resampling.shape, sep='-->')
    
      out_meta = riod11.meta
      out_meta.update({'dtype':rio.float32})
        
      fname11 = 'resampling_'+os.path.basename(riod11.name)
      with rio.open(os.path.join(s2Output, fname11), "w", **out_meta) as dest:
        dest.write(riod11.read().astype(rio.float32))
    
      fname12 = 'resampling_'+os.path.basename(riod12.name)
      with rio.open(os.path.join(s2Output, fname12), "w", **out_meta) as dest:
        dest.write(riod12.read().astype(rio.float32))
    
      fname08 = 'resampling_'+os.path.basename(riod08.name)
      with rio.open(os.path.join(s2Output, fname08), "w", **out_meta) as dest:
        dest.write(riod08_resampling.astype(rio.float32))
    
      fname04 = 'resampling_'+os.path.basename(riod04.name)
      with rio.open(os.path.join(s2Output, fname04), "w", **out_meta) as dest:
        dest.write(riod04_resampling.astype(rio.float32))
    
      fname03 = 'resampling_'+os.path.basename(riod03.name)
      with rio.open(os.path.join(s2Output, fname03), "w", **out_meta) as dest:
        dest.write(riod03_resampling.astype(rio.float32))
    
      fname02 = 'resampling_'+os.path.basename(riod02.name)
      with rio.open(os.path.join(s2Output, fname02), "w", **out_meta) as dest:
        dest.write(riod02_resampling.astype(rio.float32))

# 使用するディレクトリ
dst_dir = '/content/drive/MyDrive/satelite/s2Bands'  # Google Colabでは'/content~~'が正
os.makedirs(dst_dir, exist_ok=True)

shape_path = "/content/drive/MyDrive/satelite/ibrakiPolygon/"  # Google Colabでは'/content~~'が正
os.makedirs(shape_path, exist_ok=True)

s2Output = '/content/drive/MyDrive/satelite/s2Output'  # Google Colabでは'/content~~'が正
os.makedirs(s2Output, exist_ok=True) # outputデータ保存ディレクトリ

# 座標の基準とするラスターデータ読み込み
raster_crs = rio.open(os.path.join(dst_dir,'S2B_54SVE_20180428_0_L2A_B11.tif'))
raster_profile = raster_crs.profile

# 小地域区分のベクターデータ（from e-Stat）を読み込みcrsをラスターデータに合わせる
shape_path = "/content/drive/MyDrive/satelite/ibrakiPolygon/"  # Google Colabでは'/content~~'が正
os.makedirs(shape_path, exist_ok=True)
part_in_shape = gpd.read_file(os.path.join(shape_path, "r2ka08235.shp"), encoding="shift-jis")[['PREF_NAME', 'CITY_NAME', 'S_NAME', 'AREA', 'PERIMETER', 'JINKO', 'SETAI', 'geometry']]
re_shape_tsukuba_mirai_2RasterCrs = part_in_shape.to_crs(raster_profile["crs"])  #crs合わせ
print(re_shape_tsukuba_mirai_2RasterCrs.shape)
# 同じ小地域がさらに細かく分かれている場合があるので、小地域単位でグループ化しておく
re_shape_tsukuba_mirai_2RasterCrs = re_shape_tsukuba_mirai_2RasterCrs.dissolve(['PREF_NAME', 'CITY_NAME', 'S_NAME'], aggfunc='sum', as_index=False)
print(re_shape_tsukuba_mirai_2RasterCrs.shape)  # 特定の小地域が統合されレコード数が減る
display(re_shape_tsukuba_mirai_2RasterCrs)

# 茨城県を含む土壌分類のデータをダウンロードしておく(国土数値情報ダウンロードサイト 土地利用細分メッシュデータ4つ)
if DOWNLOAD:
    print(1)
    ground_truth = gpd.read_file(os.path.join(shape_path, "L03-b-21_5340-tky.shp"), encoding="shift-jis")
    ground_truth = ground_truth.to_crs(raster_profile["crs"])  #crs合わせ
    ground_truth = gpd.overlay(re_shape_tsukuba_mirai_2RasterCrs, ground_truth, how='intersection')  # クロップ
    print(2)
    ground_truth2 = gpd.read_file(os.path.join(shape_path, "L03-b-21_5339-tky.shp"), encoding="shift-jis")
    ground_truth2 = ground_truth2.to_crs(raster_profile["crs"])  #crs合わせ
    ground_truth2 = gpd.overlay(re_shape_tsukuba_mirai_2RasterCrs, ground_truth2, how='intersection')
    print(3)
    ground_truth3 = gpd.read_file(os.path.join(shape_path, "L03-b-21_5440-tky.shp"), encoding="shift-jis")
    ground_truth3 = ground_truth3.to_crs(raster_profile["crs"])  #crs合わせ
    ground_truth3 = gpd.overlay(re_shape_tsukuba_mirai_2RasterCrs, ground_truth3, how='intersection')
    print(4)
    ground_truth4 = gpd.read_file(os.path.join(shape_path, "L03-b-21_5439-tky.shp"), encoding="shift-jis")
    ground_truth4 = ground_truth4.to_crs(raster_profile["crs"])  #crs合わせ
    ground_truth4 = gpd.overlay(re_shape_tsukuba_mirai_2RasterCrs, ground_truth4, how='intersection')

# 4つのデータを連結
if DOWNLOAD:
    ground_truth_2RasterCrs_concat = pd.concat([ground_truth
                                                , ground_truth2
                                                , ground_truth3
                                                , ground_truth4]).reset_index(drop=True)
    ground_truth_2RasterCrs_concat.to_file(os.path.join(shape_path, "ground_truth_2RasterCrs_concat.shp"), encoding="shift-jis")

# 連結した土壌分類でデータ読み込み
ground_truth_2RasterCrs_concat = gpd.read_file(os.path.join(shape_path, "ground_truth_2RasterCrs_concat.shp"), encoding="shift-jis")
display(ground_truth_2RasterCrs_concat)

# 一部MultiPolygonの地域があるのでPolygonに紐解いておく
ground_truth_2RasterCrs_concat_crop_exploded = ground_truth_2RasterCrs_concat.explode().reset_index(drop=True)  # MultiPolygonをPolygonに解く
# 土壌分類種をラベルエンコーディングしておく
le = sklearn.preprocessing.LabelEncoder()
le_L03b_002 = le.fit_transform(ground_truth_2RasterCrs_concat_crop_exploded['L03b_002'])
ground_truth_2RasterCrs_concat_crop_exploded['le_L03b_002'] = le_L03b_002
display(ground_truth_2RasterCrs_concat_crop_exploded)

# 土壌分類のマスタ作成
'''
https://nlftp.mlit.go.jp/ksj/gml/codelist/LandUseCd-09.html

0100	田	湿田・乾田・沼田・蓮田及び田とする。
0200	その他の農用地	麦・陸稲・野菜・草地・芝地・りんご・梨・桃・ブドウ・茶・桐・はぜ・こうぞ・しゅろ等を栽培する土地とする。
0300	-	-
0400	-	-
0500	森林	多年生植物の密生している地域とする。
0600	荒地	しの地・荒地・がけ・岩・万年雪・湿地・採鉱地等で旧土地利用データが荒地であるところとする。
0700	建物用地	住宅地・市街地等で建物が密集しているところとする。
0800	-	-
0901	道路	道路などで、面的に捉えられるものとする。
0902	鉄道	鉄道・操車場などで、面的にとらえられるものとする。
1000	その他の用地	運動競技場、空港、競馬場・野球場・学校・港湾地区・人工造成地の空地等とする。
1100	河川地及び湖沼	人工湖・自然湖・池・養魚場等で平水時に常に水を湛えているところ及び河川・河川区域の河川敷とする。
1200	-	-
1300	-	-
1400	海浜	海岸に接する砂、れき、岩の区域とする。
1500	海水域	隠顕岩、干潟、シーパースも海に含める。
1600	ゴルフ場	ゴルフ場のゴルフコースの集まっている部分のフェアウエイ及びラフの外側と森林の境目を境界とする。
'''
area_use_categories = {'0100':'田', '0200':'その他の農用地', '0500':'森林', '0600':'荒地', '0700':'建物用地', '0901':'道路'
                       , '0902':'鉄道', '1000':'その他の用地', '1100':'河川地及び湖沼', '1400':'海浜', '1500':'海水域', '1600':'ゴルフ場'
                      }
area_use_categories_le = {j:area_use_categories[i] for i, j in zip(le.classes_, le.transform(le.classes_))}
display(area_use_categories)
display(area_use_categories_le)

# 試しにベクターデータを可視化
plt.rcParams['font.family'] = prop.get_name() #全体のフォントを設定
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot(1,3,1)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='None', ax=ax, edgecolor='k', linewidth=0.3)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(False)
# 一部だけグラウンドトゥルースプロット
ax = plt.subplot(1,3,2)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='None', ax=ax, edgecolor='k', linewidth=0.3)
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
ground_truth_2RasterCrs_concat_crop_exploded.sample(2000).plot(column='le_L03b_002', facecolor='pink', ax=ax, edgecolor='k', linewidth=0.05, cmap='tab10', legend=True, cax=cax)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(False)
cax.tick_params(labelsize='4')
cax.set_yticks([c for c in area_use_categories_le.keys()])
cax.set_yticklabels([c for c in area_use_categories_le.values()])
# グラウンドトゥルースプロット
ax = plt.subplot(1,3,3)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='None', ax=ax)
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
img = ground_truth_2RasterCrs_concat_crop_exploded.iloc[:,:].plot(column='le_L03b_002', facecolor='lightpink', ax=ax, edgecolor='k', linewidth=0.05, cmap='tab10', legend=True, cax=cax)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(False)
cax.tick_params(labelsize='4')
cax.set_yticks([c for c in area_use_categories_le.keys()])
cax.set_yticklabels([c for c in area_use_categories_le.values()])
plt.tight_layout()
plt.show()

# 各バンドのtifファイルのリスト
B04s_resampling = sorted(list(glob.glob(os.path.join(s2Output,'resampling_S2*B04_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B08s_resampling = sorted(list(glob.glob(os.path.join(s2Output,'resampling_S2*B08_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B11s_resampling = sorted(list(glob.glob(os.path.join(s2Output,'resampling_S2*B11_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B12s_resampling = sorted(list(glob.glob(os.path.join(s2Output,'resampling_S2*B12_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))

B03s_resampling = sorted(list(glob.glob(os.path.join(s2Output,'resampling_S2*B03_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
B02s_resampling = sorted(list(glob.glob(os.path.join(s2Output,'resampling_S2*B02_crop.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
print(B04s_resampling)
print(B08s_resampling)
print(B11s_resampling)
print(B12s_resampling)
print(B03s_resampling)
print(B02s_resampling)

# バンド演算し、可視化&TIFファイルとして保存
fig = plt.figure(figsize=(12, 16))
for num, (b04, b08, b11, b12, b03, b02) in enumerate(zip(B04s_resampling, B08s_resampling, B11s_resampling, B12s_resampling, B03s_resampling, B02s_resampling)):
    print('#####', os.path.basename(b08).split('_')[3], '#####')
    riod04_resampling = rio.open(b04).read()  # Red
    riod08_resampling = rio.open(b08).read()  # NIR
    riod11 = rio.open(b11)
    riod11_resampling = riod11.read()  # SWIR1
    riod12 = rio.open(b12)
    riod12_resampling = riod12.read()  # SWIR2
    riod03_resampling = rio.open(b03).read()  # Green
    riod02_resampling = rio.open(b02).read()  # Blue
    bounds = riod11.bounds
    print(riod04_resampling.shape)
    print(riod08_resampling.shape)
    print(riod11_resampling.shape)
    print(riod12_resampling.shape)
    print(riod03_resampling.shape)
    print(riod02_resampling.shape)

    # 可視近赤外(VNIR)、短波長赤外(SWIR)、熱赤外(TIR)、近赤外(NRI)
    # NDBI = SWIR1(Band11)-NIR(Band8) / SWIR1(Band11)+NIR(Band8)  正規化都市化指数
    # UI = ((SWIR2 - NIR) / (SWIR2 + NIR))  都市化指数
    # NDVI = NIR(Band8) - RED(Band4) / NIR(Band8) + RED(Band4)  正規化植生指数
    # GNDVI = ( NIR - Green) / ( NIR + Green)  Green正規化植生指数
    # BA = NDBI - NDVI  都市域
    # MSAVI2 = (1/2)*(2(NIR+1)-sqrt((2*NIR+1)^2-8(NIR-Red)))  修正土壌調整植生指数
      # SAVI(土壌調整係数を使用して、土壌の明るさの影響を最小限に抑えることを目的とした植生指数)での露出土壌の影響を最小限に抑えることを目的とした植生指数
    # MNDWI = (Green - SWIR) / (Green + SWIR)  修正正規化水指数
    # NDWI = (NIR-SWIR) / (NIR+SWIR)  正規化水指数
    # NDSI = (SWIR2 - Green) / (SWIR2 + Green)  正規化土壌指数
    # BSI = (SWIR1+Green)-(NIR+Blue) / (SWIR1+Green)+(NIR+Blue)  裸地化指数
    # DBSI =( (SWIR1 − GREEN) / (SWIR1 + GREEN) ) − NDVI  乾燥裸地指数
    NDBI = ( riod11_resampling.astype(float) - riod08_resampling.astype(float) ) / ( riod11_resampling.astype(float) + riod08_resampling.astype(float) )
    UI = ( riod12_resampling.astype(float) - riod08_resampling.astype(float) ) / ( riod12_resampling.astype(float) + riod08_resampling.astype(float) )
    NDVI = ( riod08_resampling.astype(float) - riod04_resampling.astype(float) ) / ( riod08_resampling.astype(float) + riod04_resampling.astype(float) )
    GNDVI = ( riod08_resampling.astype(float) - riod03_resampling.astype(float) ) / ( riod08_resampling.astype(float) + riod03_resampling.astype(float) )
    BA = NDBI - NDVI
    MSAVI2 = (1/2)*( 2*(riod08_resampling.astype(float)+1) - np.sqrt( (2*riod08_resampling.astype(float)+1)**2 - 8*(riod08_resampling.astype(float)-riod04_resampling.astype(float)) ) )
    MNDWI = (riod03_resampling.astype(float) - riod11_resampling.astype(float)) / (riod03_resampling.astype(float) + riod11_resampling.astype(float))
    NDWI = (riod04_resampling.astype(float) - riod11_resampling.astype(float)) / (riod04_resampling.astype(float) + riod11_resampling.astype(float))
    NDSI = ( riod12_resampling.astype(float) - riod03_resampling.astype(float) ) / ( riod12_resampling.astype(float) + riod03_resampling.astype(float) )
    BSI = ( (riod11_resampling.astype(float) + riod03_resampling.astype(float)) - (riod08_resampling.astype(float) + riod02_resampling.astype(float)) ) / ( (riod11_resampling.astype(float) + riod03_resampling.astype(float)) + (riod08_resampling.astype(float) + riod02_resampling.astype(float)) )
    DBSI = ( ( riod11_resampling.astype(float) - riod03_resampling.astype(float) ) / ( riod11_resampling.astype(float) + riod03_resampling.astype(float) ) ) - NDVI
    
    # NDBI可視化
    ax = plt.subplot(11,6,num+1)
    img = ax.imshow(NDBI[0], cmap='coolwarm', vmin=np.quantile(NDBI[0], q=0.01), vmax=np.quantile(NDBI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('NDBI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # UI可視化
    ax = plt.subplot(11,6,num+1+6)
    img = ax.imshow(UI[0], cmap='coolwarm', vmin=np.quantile(UI[0], q=0.01), vmax=np.quantile(UI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('UI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)
    
    # BA可視化
    ax = plt.subplot(11,6,num+1+12)
    img = ax.imshow(BA[0], cmap='RdBu_r', vmin=np.quantile(BA[0], q=0.01), vmax=np.quantile(BA[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('BA_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)
    
    # NDVI可視化
    ax = plt.subplot(11,6,num+1+18)
    img = ax.imshow(NDVI[0], cmap='RdYlGn', vmin=np.quantile(NDVI[0], q=0.01), vmax=np.quantile(NDVI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('NDVI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # GNDVI可視化
    ax = plt.subplot(11,6,num+1+24)
    img = ax.imshow(GNDVI[0], cmap='RdYlGn', vmin=np.quantile(GNDVI[0], q=0.01), vmax=np.quantile(GNDVI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('GNDVI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # MSAVI2可視化
    ax = plt.subplot(11,6,num+1+30)
    img = ax.imshow(MSAVI2[0], cmap='RdYlGn', vmin=np.quantile(MSAVI2[0], q=0.01), vmax=np.quantile(MSAVI2[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('MSAVI2_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # MNDWI可視化
    ax = plt.subplot(11,6,num+1+36)
    img = ax.imshow(MNDWI[0], cmap='Blues', vmin=np.quantile(MNDWI[0], q=0.01), vmax=np.quantile(MNDWI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('MNDWI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # NDWI可視化
    ax = plt.subplot(11,6,num+1+42)
    img = ax.imshow(NDWI[0], cmap='Blues', vmin=np.quantile(NDWI[0], q=0.01), vmax=np.quantile(NDWI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('NDWI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # NDSI可視化
    ax = plt.subplot(11,6,num+1+48)
    img = ax.imshow(NDSI[0], cmap='PuOr_r', vmin=np.quantile(NDSI[0], q=0.01), vmax=np.quantile(NDSI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('NDSI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # BSI可視化
    ax = plt.subplot(11,6,num+1+54)
    img = ax.imshow(BSI[0], cmap='PuOr_r', vmin=np.quantile(BSI[0], q=0.01), vmax=np.quantile(BSI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('BSI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)

    # DBSI可視化
    ax = plt.subplot(11,6,num+1+60)
    img = ax.imshow(DBSI[0], cmap='PuOr_r', vmin=np.quantile(DBSI[0], q=0.01), vmax=np.quantile(DBSI[0], q=0.99), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=4)
    ax.set_title('DBSI_'+os.path.basename(riod11.name).split('_')[3]+'_'+os.path.basename(riod11.name).split('_')[6], fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.5, ax=ax, linewidth=0.5)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)
    
    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'NDBI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(NDBI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'UI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(UI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'BA_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(BA.astype(rio.float32))
    
    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'NDVI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(NDVI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'GNDVI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(GNDVI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'MSAVI2_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(MSAVI2.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'MNDWI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(MNDWI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'NDWI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(NDWI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'NDSI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(NDSI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'BSI_'+os.path.basename(riod11.name)
    print(fname)
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(BSI.astype(rio.float32))

    out_meta = riod11.meta
    out_meta.update({'dtype':rio.float32})
    fname = 'DBSI_'+os.path.basename(riod11.name)
    print(fname,'\n')
    with rio.open(os.path.join(s2Output, fname), "w", **out_meta) as dest:
        dest.write(DBSI.astype(rio.float32))

plt.tight_layout()
plt.show()

# 各バンドのtifファイルのリスト
NDBI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'NDBI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
UI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'UI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
BA_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'BA_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
NDVI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'NDVI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
GNDVI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'GNDVI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
MSAVI2_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'MSAVI2_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
MNDWI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'MNDWI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
NDWI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'NDWI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
NDSI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'NDSI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
BSI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'BSI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
DBSI_band_path_list_nomask = sorted(list(glob.glob(os.path.join(s2Output,'DBSI_resampling_S2*.tif*'))), key=lambda x:(x.split('_54SVE_')[-1]))
band_paths_list = NDBI_band_path_list_nomask + UI_band_path_list_nomask + BA_band_path_list_nomask + NDVI_band_path_list_nomask + GNDVI_band_path_list_nomask + MSAVI2_band_path_list_nomask + MNDWI_band_path_list_nomask + NDWI_band_path_list_nomask + NDSI_band_path_list_nomask + BSI_band_path_list_nomask + DBSI_band_path_list_nomask
rio_file_list2020 = band_paths_list[2:][::6]
display(rio_file_list2020)

# 土壌分類のベクターファイルのPolygon内のバンド演算指標の平均値を計算して、TIFファイルをベクターファイル化
def add_stats_vals(rio_data, vec_data, add_col_name, area_col='S_NAME'):
    affine = rio_data.transform
    mean_stats = rasterstats.zonal_stats(vec_data, rio_data.read(1)
                                         , affine=affine
                                         , add_stats={'mymean':lambda x: np.ma.mean(x)}) # np.ma.filled(x, fill_value=np.nan)
    mean_stats = [m['mymean'] for m in mean_stats]  # 平均値を取得
    vec_data_stats = vec_data#.copy()
    vec_data_stats[add_col_name] = mean_stats
    vec_data_stats[add_col_name] = vec_data_stats[add_col_name].map(lambda x: np.ma.filled(x, fill_value=np.nan))  # Maskの箇所はnanに変更
    # 欠損値は小地域ごとの平均値で埋める
    category_means = vec_data_stats.dropna(subset=[add_col_name]).groupby(area_col)[add_col_name].mean()  # カテゴリごとの平均を計算
    vec_data_stats[add_col_name] = vec_data_stats[add_col_name].fillna(vec_data_stats[area_col].map(category_means))
    return vec_data_stats

if DOWNLOAD:
    ground_truth_2RasterCrs_concat_crop_exploded_stats = ground_truth_2RasterCrs_concat_crop_exploded.copy()
    for riofile in rio_file_list2020:
        riodata = rio.open(riofile)
        ground_truth_2RasterCrs_concat_crop_exploded_stats = add_stats_vals(riodata
                                                                            , ground_truth_2RasterCrs_concat_crop_exploded_stats
                                                                            , os.path.basename(riofile).split('_')[0]+'_mean'
                                                                            , area_col='S_NAME')
    ground_truth_2RasterCrs_concat_crop_exploded_stats.to_file(os.path.join(shape_path, "ground_truth_2RasterCrs_concat_crop_exploded_stats.shp"), encoding="shift-jis")
    display(ground_truth_2RasterCrs_concat_crop_exploded_stats)

# SHAPEファイルのカラム名は10文字以内の制限があるので、"MSAVI2_mean"は"MSAVI2_mea"になってしまっている
ground_truth_2RasterCrs_concat_crop_exploded_stats = gpd.read_file(os.path.join(shape_path, "ground_truth_2RasterCrs_concat_crop_exploded_stats.shp"), encoding="shift-jis")
ground_truth_2RasterCrs_concat_crop_exploded_stats.rename(columns={'MSAVI2_mea':'MSAVI2_mean'}, inplace=True)
ground_truth_2RasterCrs_concat_crop_exploded_stats.rename(columns={'le_L03b_00':'le_L03b_002'}, inplace=True)
display(ground_truth_2RasterCrs_concat_crop_exploded_stats)

# PolygonをPointにする
ground_truth_2RasterCrs_concat_crop_exploded_stats_2point = ground_truth_2RasterCrs_concat_crop_exploded_stats.copy()
ground_truth_2RasterCrs_concat_crop_exploded_stats_2point['geometry'] = [p.centroid for p in ground_truth_2RasterCrs_concat_crop_exploded_stats.geometry]

# Point可視化（一例としてNDSIのみ）
satIndex='NDSI'
cmap='PuOr_r'
fig = plt.figure(figsize=(3, 3))
ax = plt.subplot(1,1,1)
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
img = ground_truth_2RasterCrs_concat_crop_exploded_stats_2point.plot(column=satIndex+'_mean', cmap=cmap, edgecolor='k', legend=True, ax=ax, cax=cax
                                                                     , linewidth=0.05, markersize=2
                                                                     , vmin=np.quantile(ground_truth_2RasterCrs_concat_crop_exploded_stats[satIndex+'_mean'], 0.01)
                                                                     , vmax=np.quantile(ground_truth_2RasterCrs_concat_crop_exploded_stats[satIndex+'_mean'], 0.99)
                                                                    )
cax.tick_params(labelsize='4', width=0.4, length=5)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.set_title(satIndex, fontsize=4)
# re_shape_tsukuba_mirai_2RasterCrs_NDBI.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(False)
plt.show()

# 説明変数
explanatory_variables = ground_truth_2RasterCrs_concat_crop_exploded_stats_2point.filter(like='mean', axis=1).columns.to_list()
# 目的変数
objective_variables = 'le_L03b_002'

# split data
X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(ground_truth_2RasterCrs_concat_crop_exploded_stats_2point[explanatory_variables+['geometry']]
                                                                         , ground_truth_2RasterCrs_concat_crop_exploded_stats_2point[[objective_variables]+['geometry']]
                                                                         , test_size=0.92, shuffle=True, random_state=0
                                                                         , stratify=ground_truth_2RasterCrs_concat_crop_exploded_stats_2point[objective_variables])
print(X_train.shape, X_test.shape)
# 各土壌分類のレコード数
display(y_train[objective_variables].value_counts())
display(y_test[objective_variables].value_counts())

# 学習データとテストデータの土壌分類
plt.rcParams['font.family'] = prop.get_name() #全体のフォントを設定
fig = plt.figure(figsize=(6, 3))
ax = plt.subplot(1,2,1)
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
img = y_train.plot(column=objective_variables, cmap='tab10', edgecolor='k', legend=True, ax=ax, cax=cax, linewidth=0.05, markersize=5)
cax.tick_params(labelsize='4', width=0.4, length=5)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.set_title('土壌分類 Point Train', fontsize=4)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cax.set_yticks([c for c in area_use_categories_le.keys()])
cax.set_yticklabels([c for c in area_use_categories_le.values()])
ax.grid(False)

ax = plt.subplot(1,2,2)
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
img = y_test.plot(column=objective_variables, cmap='tab10', edgecolor='k', legend=True, ax=ax, cax=cax, linewidth=0.05, markersize=1)
cax.tick_params(labelsize='4', width=0.4, length=5)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.set_title('土壌分類 Point Test', fontsize=4)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cax.set_yticks([c for c in area_use_categories_le.keys()])
cax.set_yticklabels([c for c in area_use_categories_le.values()])
ax.grid(False)
plt.tight_layout()
plt.show()
```

</details>

# モデル構築前の準備
## sample weightの定義
土壌クラスは10種あるので、構築するモデルは多項分類モデルとなる。  
その土壌クラスのデータ数が不均衡なので、以前の記事（[「衛星データでつくばみらい市の土壌分類をしてみた（地理空間データ解析）」](https://qiita.com/chicken_data_analyst/items/886d35561a4f23653dc4)）のように不均衡データへの対応としてsample weightを設定する。
```python
# sample weightの定義
def sample_w(y_train):
    '''
    output sample weight (balanced weight)
    y_train:True Train data
    '''
    n_samples=len(y_train)
    n_classes=len(np.unique(y_train))
    bincounts = {i:len(y_train[y_train==i]) for i in sorted(np.unique(y_train))}
    class_ratio_param = {key:n_samples / (n_classes * bincnt) for key, bincnt in bincounts.items()}
    #print('class_ratio_param',class_ratio_param)
    sample_weight=np.array([class_ratio_param[r] for r in y_train])
    return sample_weight
    
# 重みの配列
sample_ws = sample_w(y_train[objective_variables])
```

## 関数定義
可視化の関数を定義しておく。

<details><summary>可視化の関数(折り畳み)</summary>

```python
# 実測データ、予測データ、分類成功可否を可視化
def point_plot(results, re_shape_tsukuba_mirai_2RasterCrs, linewidth=0.05, markersize=5, title='', vmin=-8, vmax=8):
    plt.rcParams['font.family'] = prop.get_name() #全体のフォントを設定
    fig = plt.figure(figsize=(9, 3))
    ax = plt.subplot(1,3,1)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    img = results.plot(column=objective_variables, cmap='tab10', edgecolor='k', legend=True, ax=ax, cax=cax, linewidth=linewidth, markersize=markersize)
    cax.tick_params(labelsize='4', width=0.4, length=5)
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    ax.set_title(title+' Point True', fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)
    cax.set_yticks([c for c in area_use_categories_le.keys()])
    cax.set_yticklabels([c for c in area_use_categories_le.values()])
    
    ax = plt.subplot(1,3,2)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    img = results.plot(column='pred', cmap='tab10', edgecolor='k', legend=True, ax=ax, cax=cax, linewidth=linewidth, markersize=markersize)
    cax.tick_params(labelsize='4', width=0.4, length=5)
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    ax.set_title(title+' Point Pred', fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)
    cax.set_yticks([c for c in area_use_categories_le.keys()])
    cax.set_yticklabels([c for c in area_use_categories_le.values()])
    
    ax = plt.subplot(1,3,3)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    img = results.plot(column='diff', cmap='binary', edgecolor='k', legend=True, ax=ax, cax=cax, linewidth=linewidth, markersize=markersize, vmin=vmin, vmax=vmax)
    cax.tick_params(labelsize='4', width=0.4, length=5)
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=4)
    ax.set_title(title+' Point False:1, True:0', fontsize=4)
    re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
    ax.yaxis.offsetText.set_fontsize(4)
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# 空間相関変数の可視化
def spatial_pattern(ground_truth, z_cars_arr_test, X_test, classK, line_vector, area_use_categories_le, node_name='z_car', figsize=(8,6), subplot_num=(3,4), markersize=1):
    plt.rcParams['font.family'] = prop.get_name()
    fig=plt.figure(figsize=figsize)
    for k, node in enumerate([node_name+str(i+1) for i in classK]):
        plot_icar = ground_truth.loc[X_test.index,:].copy()
        icar = z_cars_arr_test[:,k]
        plot_icar[node] = icar
        ax = plt.subplot(subplot_num[0],subplot_num[1],k+1)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')
        img = plot_icar.plot(column=node, cmap='RdYlBu_r', edgecolor='k', legend=True, ax=ax, cax=cax, linewidth=0.05, markersize=markersize
                             #, vmin=np.quantile(plot_icar[node], q=0.01), vmax=np.quantile(plot_icar[node], q=0.99)
                            )
        cax.tick_params(labelsize='4', width=0.4, length=5)
        plt.setp(ax.get_xticklabels(), fontsize=4)
        plt.setp(ax.get_yticklabels(), fontsize=4)
        ax.set_title(node+' '+area_use_categories_le[k], fontsize=4)
        line_vector.plot(facecolor='none', edgecolor='k', alpha=0.8, ax=ax, linewidth=0.2)
        ax.yaxis.offsetText.set_fontsize(4)
        ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.grid(False)
    plt.tight_layout()
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
```

</details>

## 隣接行列の定義
CARモデルの構築には隣接行列が必要なので定義する。  
最近隣4地域を隣接地域として計算する。  
`libpysal.cg.KDTree`と`libpysal.weights.KNN`を使って隣接行列を作り可視化する。  
```python
# 最近隣4ゾーンの計算して可視化する
# 重心（中心点）の計算
coords = X_train.centroid.map(lambda geom: (geom.x, geom.y)).tolist()
kd = libpysal.cg.KDTree(np.array(coords))
# 最近隣4ゾーンの計算
wnn2 = libpysal.weights.KNN(kd, 4)
# GeoDataFrameのプロット
fig = plt.figure(figsize=(2, 2))
ax = plt.subplot(1,1,1)
re_shape_tsukuba_mirai_2RasterCrs.plot(facecolor='none', edgecolor='k', ax=ax, linewidth=0.2)
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.setp(ax.get_yticklabels(), fontsize=4)
ax.set_title('最近隣4ゾーン', fontsize=4)
ax.yaxis.offsetText.set_fontsize(4)
ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 隣接行列のプロット
for i, (key, val) in enumerate(wnn2.neighbors.items()):
    for neighbor in val:
        ax.plot([coords[i][0], coords[neighbor][0]], [coords[i][1], coords[neighbor][1]], color='red', linewidth=0.1
                #, marker='+', markersize=0.001, markerfacecolor="k", markeredgecolor="k"
               )
ax.grid(False)
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/766e7433-8e7c-3740-17af-091192ba238e.png)

ここで作った隣接行列をこの後にそのまま使うと非対称な隣接行列はダメだってエラーがでた。つまり、地域Aは地域Bと隣接なのに、地域Bは地域Aと隣接ではないという場合があって非対称となってしまっているそうだ。そのため片方が隣接ならもう片方も隣接とするような処理を実施した。（なので地域によっては近隣地域が5以上の場合もあるだろう。）
```python
# 隣接行列作成
adj = wnn2.to_adjlist()
adj_symmetry = np.zeros((len(y_train), len(y_train)))
for i, (focal, neighbor) in tqdm(enumerate(zip(adj['focal'],adj['neighbor']))):
    if (focal, neighbor) in wnn2.asymmetry():
        # 非対称の場合、お互いに1とする
        adj_symmetry[focal, neighbor] = adj_symmetry[neighbor, focal] = 1
        continue
    adj_symmetry[focal, neighbor] = adj_symmetry[neighbor, focal] = 1
print(adj_symmetry.shape)
# >> (851, 851)
```

各地域とその隣接のペアの配列を定義しておく。隣接行列からそのままペアを作ると、A-B,B-Aのような重複したペアができてしまうので隣接行列の上三角行列（もしくは下三角行列）を取得してペアを抽出する。  
これは[[Github]ICAR class](https://www.pymc.io/projects/docs/en/v5.13.0/_modules/pymc/distributions/multivariate.html#ICAR)を参考にした。
```python
# 上三角行列から近接ペア取得（A-B,B-Aのように重複しないように上三角行列を使用）
node1, node2 = np.where(np.triu(adj_symmetry) == 1)
print(node1.shape, node2.shape)
# >> (2057,) (2057,)
```

土壌分類の数とエンコード済みの土壌分類の配列を定義しておく。
```python
# 土壌分類コードリスト
classK = sorted(y_train[objective_variables].unique())
K = len(classK)
print(classK)
print(K)
# >> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# >> 10
```

# Lerouxモデル
## Intrinsic CARモデルのおさらい
まずCARモデルの説明を簡単にしていく。（参考：[[書籍]Rではじめる地理空間データの統計解析入門](https://www.kspub.co.jp/book/detail/5273036.html)）  

CARモデルの最も単純な形は地域特性$z_{i}$と偶発的な要因などを吸収するための誤差項$\epsilon_{i}$だけのモデル。
```math
\displaylines{
y_{i}=z_{i}+\epsilon_{i}
\\
\epsilon_{i} \sim \mathcal{N}(0,\sigma^2)
}
```
$z_{i}$をどのようにモデル化するかでCARモデルにはバリエーションがあるが、代表的なものにICAR（Intrinsic CAR）モデルがある。このモデルでは$z_{i}$は以下のように定義される。
```math
z_{i}|z_{j{\neq}i} \sim \mathcal{N}(\frac{\sum_{j} \omega_{ij}z_{j}}{\sum_{j} \omega_{ij}}, \frac{\tau^{2}}{\sum_{j} \omega_{ij}})
```
$z_{i}|z_{j{\neq}i}$は地域$i$以外の地域における変数$z_{j{\neq}i}$で条件づけられた変数$z_{i}$で、他の地域$z_{j{\neq}i}$の値がすでに分かっていると仮定して空間相関変数$z_{i}$をモデル化している。  
$z_{i}|z_{j{\neq}i}$の期待値は隣接行列$\omega_{ij}$を使って$\frac{\sum_{j} \omega_{ij}z_{j}}{\sum_{j} \omega_{ij}}$と表され、隣接地域との空間相関を考慮していると言える。$\tau^{2}$は空間相関変数$z_{i}$の分散で、$\tau^{2}$が大きいと空間相関で説明される変動が大きくなり、$\tau^{2}=0$だと、空間相関パターンは消失する。  
$\epsilon_{i}$の分散$\sigma^2$と$z_{i}$の分散$\tau^{2}$で目的変数の総変動に占める空間相関成分の割合を表すと$\frac{\tau^{2}}{\sigma^2+\tau^{2}}$となり、これが1に近いと空間相関成分の割合が大きいと言える。  

説明変数も考慮したICARモデルは以下のように表すことができる。
```math
\displaylines{
y_{i}=\sum_{k=1}^{K}x_{i,k}\beta_{k}+z_{i}+\epsilon_{i},
\\
z_{i}|z_{j{\neq}i} \sim \mathcal{N}(\frac{\sum_{j} \omega_{ij}z_{j}}{\sum_{j} \omega_{ij}}, \frac{\tau^{2}}{\sum_{j} \omega_{ij}}),
\\
\epsilon_{i} \sim \mathcal{N}(0,\sigma^2)
}
```
## Lerouxモデルの概要
※参考：[[書籍]Rではじめる地理空間データの統計解析入門](https://www.kspub.co.jp/book/detail/5273036.html)  

ICARモデルでは弱い空間相関を捉えられないという短所がある。ICARでは自分の地域の事前分布の平均は$\frac{\sum_{j} \omega_{ij}z_{j}}{\sum_{j} \omega_{ij}}$で与えられるので、例えば隣接地域の数値が高ければ、自分の地域の数値の期待値は必ず上がる。しかし実際は自分の地域の数値はある程度高いものの隣接地域程ではない可能性もある。このような弱い相関を捉えるためにICARモデルの拡張モデルがいくつかある。  
Lerouxモデルはその中の1つで、地域$i$以外の地域$j \neq i$で条件づけられた空間相関変数$z_{i}|z_{j{\neq}i}$の分布は以下のように与えらる。
```math
\begin{equation}
\begin{split}

z_{i}|z_{j{\neq}i} &\sim \mathcal{N}(\frac{\rho \sum_{j} \omega_{ij}z_{j}}{\rho \sum_{j} \omega_{ij}+1- \rho }, \frac{\tau^{2}}{\rho \sum_{j} \omega_{ij}+1- \rho }) \\

\end{split}
\end{equation}
```

空間相関の強さを表すパラメータ$\rho$を導入することで弱い空間相関も捉えようというアプローチで、$\rho=1$の時ICARモデルと一致する。$\rho=0$の時は空間相関は存在しないというモデルになる。  

ここで、[「Spatial Models in Stan: Intrinsic Auto-Regressive Models for Areal Data」](https://mc-stan.org/users/documentation/case-studies/icar_stan.html)より、Lerouxモデルを表現していく。  
まずICARモデル（$\tau=1$）を以下のように示す。  
（※$\sum_{i \sim j} \phi_{j}=\sum_{j} \omega_{ij}z_{j}$、 $i \sim j$は$i$と$j$が近傍を示す。）

```math
\begin{equation}
\begin{split}

p(\phi_{i}|\phi_{j{\neq}i}) &\sim \mathcal{N}(\frac{\sum_{i \sim j} \phi_{j}}{\sum_{j} \omega_{ij}}, \frac{1}{\sum_{j} \omega_{ij}}) \\

p(\phi) &\propto \exp \{ -\frac{1}{2}\phi^{T}[D-W]\phi \} \\

\end{split}
\end{equation}
```
$D-W$は$\tau=1$の時の$\phi$の精度行列であり、対数を取って式変形をすると、以下のようになる。

```math
\log p(\phi_) =  -\frac{1}{2}\sum_{i \sim j}(\phi_{i}-\phi_{j})^{2}+const
```

ICARではこの対数確率密度が最大になるように計算していき$\phi$の事後分布を得る。（[PyMCの`pm.ICAR`のlogp関数](https://www.pymc.io/projects/docs/en/v5.10.1/_modules/pymc/distributions/multivariate.html#ICAR)を参照）  

ここで、Lerouxモデルの精度行列は$\rho (D-W)+(1-\rho)I$と表される（[「Investigation of Bayesian spatial models」](https://cancerqld.blob.core.windows.net/content/docs/Investigation-of-Bayesian-spatial-models.pdf)）ことを考えると、Lerouxモデルの$\log p(\phi_)$は、

```math
\begin{equation}
\begin{split}

\log p(\phi_) &=  -\frac{1}{2} \{ \rho \sum_{i \sim j}(\phi_{i}-\phi_{j})^{2} + (1-\rho)\phi^{T}I\phi \}+const \\

&=  -\frac{1}{2} \{ \rho \sum_{i \sim j}(\phi_{i}-\phi_{j})^{2} + (1-\rho)\sum_{i}\phi_{i}^2 \}+const

\end{split}
\end{equation}
```

と表されるので、この対数確率密度をもとに$\phi$の事後分布を得ることでLerouxモデルを構築することができるはず。

## Lerouxモデル構築
[PyMCの`pm.ICAR`のlogp関数](https://www.pymc.io/projects/docs/en/v5.10.1/_modules/pymc/distributions/multivariate.html#ICAR)を参考に、Lerouxモデルのlogpを定義し、`pm.CustomDist`を使って、モデルを構築する。
Lerouxモデルの空間相関変数の対数確率密度として`leroux_logp`を定義。$-\frac{1}{2} \{ \rho \sum_{i \sim j}(\phi_{i}-\phi_{j})^{2} + (1-\rho)\sum_{i}\phi_{i}^2 \}$に加え、$\sum_{i}\phi_{i}=0$という制約を加えるため、二乗ペナルティ項を追加している。  
`pm.CustomDist`で`ICAR10s`を定義したら後は普通のICARモデルと同じ。
```python
# Lerouxモデルの空間相関変数のlogp
def leroux_logp(value, N, node1, node2, rho):
    # logp
    pairwise_difference = (-1/2) * ( rho*pt.sum(pt.square(value[node1] - value[node2])) + (1-rho)*pt.sum(pt.square(value)) )
    # zero_sum_stdev=0.001
    # zero_sum = (-0.5 * pt.pow(pt.sum(value) / (zero_sum_stdev * N), 2) - pt.log(pt.sqrt(2.0 * np.pi)) - pt.log(zero_sum_stdev * N))
    # ICAR = pairwise_difference + zero_sum

    # sum(Φ)=0という制約が必要なので、sum(Φ)が0から離れるとペナルティが課されるようにする
    def penalty(x, threshold, penalty_coef=1):
        return penalty_coef * -((x - threshold)**2)  # 二乗ペナルティ
    zero_sum_penalty = penalty(pt.sum(value), 0)  # pt.sum(value)の0からの偏差に対するペナルティ項
    ICAR = pairwise_difference + zero_sum_penalty
    return ICAR

# Lerouxモデル
with pm.Model() as model_icar_mine:
    # coords(次元やインデックスを定義)
    model_icar_mine.add_coord('data', values=range(X_train.shape[0]), mutable=True)
    model_icar_mine.add_coord('var', values=explanatory_variables, mutable=True)
    model_icar_mine.add_coord('obj_var', values=sorted(y_train[objective_variables].unique()), mutable=True)
    
    # 変数
    x = pm.MutableData('x', X_train[explanatory_variables].to_numpy(), dims=('data', 'var'))
    y = pm.MutableData("y", y_train[objective_variables].to_numpy(), dims=('data', ))
    weights = pm.MutableData("weights", sample_ws, dims=('data', ))
    print('x shape', x.eval().shape)
    print('y shape', y.eval().shape)
    print('weights shape', weights.eval().shape)

    N = len(y.eval())  # 地域数
    rho = pm.Uniform('rho', lower=0, upper=1, dims=("obj_var", ))  # 空間相関パラメータ
    #print('rho shape', rho.eval().shape)

    # 隣接地域のペアnode1-node2
    nd1 = pt.as_tensor_variable(node1, dtype=int)
    nd2 = pt.as_tensor_variable(node2, dtype=int)
    print('nd1 shape', nd1.eval().shape)
    print('nd2 shape', nd2.eval().shape)
    # クラスの数分の空間相関変数を定義
    ICAR10s = [pm.CustomDist('z_car'+str(i+1), N, nd1, nd2, rho[i], logp=leroux_logp, dims=('data'), ndims_params=[0,1,1,0]).reshape((N, 1)) for i, k in enumerate(classK)]
    ICAR10s = pm.math.concatenate(ICAR10s, axis=1)
    tau2 = pm.Uniform('tau2', lower=0, upper=100)   # spatial var
    print('ICAR10s shape', ICAR10s.type)
    print('tau2 shape', tau2.eval().shape)

    # 推論パラメータの事前分布
    coef_ = pm.Normal('coef', mu=0.0, sigma=1, dims=("var",'obj_var'))  # 各係数の事前分布は正規分布
    intercept_ = pm.Normal('intercept', mu=0.0, sigma=1.0, dims=("obj_var", ))  # 切片の事前分布は正規分布
    print('coef shape', coef_.eval().shape)
    print('intercept shape', intercept_.eval().shape)
    
    # linear model --> 𝑦 = coef_𝑥 + intercept_ + eps_ + ICARs
    print('x.dot(coef_) shape', x.dot(coef_).eval().shape)
    print('ICAR10s*tau type', (ICAR10s*tau2).type)
    theta = pm.Deterministic("theta", pm.math.softmax(x.dot(coef_)+intercept_+(ICAR10s*tau2), axis=1), dims=('data', 'obj_var'))  # axis設定しないとダメ
    #y_pred = pm.Categorical("y_pred", p=theta, observed=y, dims='data')
    y_pred = pm.Potential('y_pred', (weights * pm.logp(pm.Categorical.dist(p=theta), y)).sum(axis=0), dims=('data', ))

# モデル構造
modeldag = pm.model_to_graphviz(model_icar_mine)
display(modeldag)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/73e2fb24-0629-0dd3-bca5-a915b9e0523b.png)

MCMC実行。
```python
%%time
# MCMC実行
# nutpieで実行
with model_icar_mine:
    # MCMCによる推論
    trace = pm.sample(draws=3000, tune=1000, chains=3, nuts_sampler="nutpie", cores=3, random_seed=1, return_inferencedata=True)

# 保存
model_dir = '/content/drive/MyDrive/satelite/model'
os.makedirs(model_dir, exist_ok=True)
# データの保存 to_netcdfの利用
trace.to_netcdf(os.path.join(model_dir, 'model_leroux.nc'))

# Load
model_dir = '/content/drive/MyDrive/satelite/model'
os.makedirs(model_dir, exist_ok=True)
# データの読み込み from_netcdfの利用
trace = az.from_netcdf(os.path.join(model_dir, 'model_leroux.nc'))
```

trace plotで収束を確認。  
あれー、$\rho$の事後分布がほぼ0だな…。合ってるのか、これ…。
まあでも収束はしてそう。
```python
# plot_trace
az.plot_trace(trace, backend_kwargs={"constrained_layout":True}, var_names=["coef","intercept","rho","tau2"])#+['z_car'+str(i) for i in range(1,11)])
plt.show()
```
![image-1.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/02908a06-46ca-824c-d6fc-7c3c30ddbd5f.png)

$\hat{R}$でも収束の確認。  
問題なさそう。
```python
# MCMCの収束を評価
rhat_vals = az.rhat(trace).values()

# 最大のRhatを確認
result = np.nanmax([np.max(i.values) for i in rhat_vals])# if i.name in ["coef","intercept",'z_car','mu','theta','tau']])
print('Max rhat:', result)
# 1.1以上のRhatを確認
for i in rhat_vals:
    if np.max(i.values)>=1.1 or np.isnan(np.max(i.values)):
        print(i.name, np.max(i.values), np.mean(i.values), i.values.shape, sep='  ====>  ')
# >> Max rhat: 1.0037927765874444
```

各クラスごとの空間相関パラメータ$\rho$を確認。  
ほげぇ…ほとんど0やんけ…。
```python
# クラスごとの空間相関
spatial_corr = {area_use_categories_le[c]:round(r,6) for c, r in enumerate(trace['posterior']['rho'].mean(dim=["chain", "draw"]).values)}
display(spatial_corr)
```
![image-24.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/b75891e8-3e97-0bf9-303c-8ed048d58d8e.png)

とりあえず分類精度とか見ていく。  
学習データの精度は悪くない。
```python
# 各クラスの事後確率theta
softmax_result = pd.DataFrame(trace['posterior']['theta'].mean(dim=["chain", "draw"]).values)
y_pred = softmax_result.idxmax(axis=1)
print(sklearn.metrics.classification_report(y_train[objective_variables].to_numpy(), y_pred.ravel()))
cm = sklearn.metrics.confusion_matrix(y_train[objective_variables].to_numpy(), y_pred.ravel())
print(cm)
```
![image-3.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/53781c9f-6994-665f-8260-73112b7770b8.png)

[前回の記事](https://qiita.com/drafts/886d35561a4f23653dc4/edit)のようにテストデータも見ていく。  
```python
# 学習データ、テストデータの座標を取得
train_coord = pd.DataFrame({'x':X_train['geometry'].x.to_list(), 'y':X_train['geometry'].y.to_list()})
test_coord = pd.DataFrame({'x':X_test['geometry'].x.to_list(), 'y':X_test['geometry'].y.to_list()})
# 学習データでknn作成
knn = sklearn.neighbors.NearestNeighbors(n_neighbors=5)
knn.fit(train_coord)
# テストデータの各サンプルと最も近い学習データのindexを取得
distances, indices = knn.kneighbors(test_coord)
test_coord['indices'] = indices[:,0]

# テストデータと最も近い学習データの空間相関変数を取得する
z_cars_arr = np.concatenate([trace['posterior']['z_car'+str(i+1)].mean(dim=["chain", "draw"]).values.reshape(-1,1) for i, k in enumerate(classK)], axis=1)  # 学習データの空間相関変数の計算
z_cars_arr_test = z_cars_arr[list(indices[:,0]),:]  # テストデータと最も近い学習データの空間相関を各テストデータに対して計算
tau_arr = trace['posterior']['tau2'].mean(dim=["chain", "draw"]).values
z_cars_arr_test = z_cars_arr_test * tau_arr  # 分散をかける
print(z_cars_arr_test.shape)
# >> (9794, 10)

# 未知データへの適用
# 回帰係数の推定値
coefs = trace['posterior']['coef'].mean(dim=["chain", "draw"]).values
# 切片の推定値
intercepts = trace['posterior']['intercept'].mean(dim=["chain", "draw"]).values
# 線形モデル式
mu = X_test[explanatory_variables].to_numpy().dot(coefs) + intercepts + z_cars_arr_test  # 回帰式（係数と切片）と空間相関変数
# ソフトマックス関数に入れる
m = softmax(mu, axis=1)  # ソフトマックス関数に入れて一般化線形モデルへ
# 確率が最大のクラスを取得
m_class = m.argmax(axis=1)  # 最も確率が高いクラスを所属クラスとする
# 精度計算
print(sklearn.metrics.classification_report(y_test[objective_variables].to_numpy(), m_class.ravel()))
cm = sklearn.metrics.confusion_matrix(y_test[objective_variables].to_numpy(), m_class.ravel())
print(cm)
```
[前回の記事](https://qiita.com/drafts/886d35561a4f23653dc4/edit)のICARと同程度。
![image-4.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/668d9f9c-eb80-5f90-abac-5173761e06b7.png)

分類結果確認。
```python
# 可視化
# 学習データ
results = y_train.copy()
results['pred'] = y_pred.ravel()
results['diff'] = results[objective_variables]-results['pred']
results.loc[(results['diff']==0), 'diff'] = 0
results.loc[(results['diff']!=0), 'diff'] = 1
point_plot(results, re_shape_tsukuba_mirai_2RasterCrs, linewidth=0.05, markersize=5, title='土壌分類', vmin=0, vmax=1)

# テストデータ
results = y_test.copy()
results['pred'] = m_class.ravel()
results['diff'] = results[objective_variables]-results['pred']
results.loc[(results['diff']==0), 'diff'] = 0
results.loc[(results['diff']!=0), 'diff'] = 1
point_plot(results, re_shape_tsukuba_mirai_2RasterCrs, linewidth=0.05, markersize=1, title='土壌分類', vmin=0, vmax=1)
```
学習データ
![image-9.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e80bc78d-778a-62bf-47e4-ed987a0d2b4f.png)
テストデータ
![image-10.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9fb0cfde-fd3f-6a3f-927a-a12b60bc421b.png)

空間パターンを見るため、空間相関変数を可視化。
```python
# 学習データの空間相関変数を可視化
spatial_pattern(ground_truth_2RasterCrs_concat_crop_exploded_stats_2point
                , z_cars_arr
                , X_train
                , classK
                , re_shape_tsukuba_mirai_2RasterCrs
                , area_use_categories_le
                , node_name='z_car', figsize=(8,6), subplot_num=(3,4), markersize=3)
# テストデータの空間相関変数（最も近い学習データの空間相関変数を取得した結果）を可視化
spatial_pattern(ground_truth_2RasterCrs_concat_crop_exploded_stats_2point
                , z_cars_arr_test
                , X_test
                , classK
                , re_shape_tsukuba_mirai_2RasterCrs
                , area_use_categories_le
                , node_name='z_car', figsize=(8,6), subplot_num=(3,4), markersize=1)
```
学習データ
![image-5.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f06a5821-5499-cf23-bcec-31bb5e724eec.png)
テストデータ
![image-6.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/205cfccc-ed9c-b57f-3490-f0e7bafeeab7.png)

うーむ。空間相関がほぼ無いモデルになったので、例えば水田のクラスの空間パターンを見ると、水田と分類したところ以外の空間相関変数はほとんど0が多い。空間相関がある場合、水田と隣接しているけど、実際は水田ではないところも空間相関変数はある程度大きくなるはずだが、その傾向が見られない。  
モデルの構築間違っていたかな…。  
$\rho$を固定して、$\rho=1$と$\rho=0.5$の2パターンの結果を見てみると以下のようになった。  

- $\rho=1$
![image-7.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/350ae3f9-82a5-60e9-7b68-0789b7030947.png)
- $\rho=0.5$
![image-8.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/400e6cbd-4d00-278c-9fc5-72ea1a8d303a.png)

$\rho=1$だと、例えば荒地を見ると荒地から離れると滑らかに空間相関変数が減衰するようなパターンが可視化されている。  
一方$\rho=0.5$だと、荒地から離れるとすぐに空間相関変数が小さくなり広い空間パターンは見られなくなる。  
ということは$\rho$の値は効いているということか…。  
$\rho$も含めて推定すると、ほとんど0になってしまうのは謎だな…。先に言ってしまうと、この後のBYM2モデルではクラスによって$\rho$が異なり、$\rho$が1に近いクラスも存在していてそっちの方が妥当に感じる。

# BYM2モデル
## BYMモデルの概要
※参考：[[書籍]Rではじめる地理空間データの統計解析入門](https://www.kspub.co.jp/book/detail/5273036.html)  

BYMはICARを拡張したモデルであり、BYM2はBYMを修正したものである。  
BYM（Besag-York-Mollie）モデルは、空間相関成分に加え、ノイズ$\theta_{j}$成分を考慮することで弱い空間相関も捉えようとするもので、以下のように表すことができる。

```math
\begin{equation}
\begin{split}

z_{i}|z_{j{\neq}i} &\sim \mathcal{N}(\frac{\sum_{j} \omega_{ij}(z_{j}+\theta_{j})}{\sum_{j} \omega_{ij}}, \frac{\tau^{2}}{\sum_{j} \omega_{ij}}) \\

\theta_{j} &\sim \mathcal{N}(0,s^2)

\end{split}
\end{equation}
```
ノイズ分散$s^2$が0の時、ICARモデルと一致する。

## BYM2モデルの概要
※参考：[A Bayesian hierarchical model for disease mapping that accounts for scaling and heavy-tailed latent effects](https://arxiv.org/pdf/2109.10330.pdf)  
※参考：[The Besag-York-Mollie Model for Spatial Data](https://www.pymc.io/projects/examples/en/latest/spatial/nyc_bym.html)  

BYMモデルでは、各地域の潜在効果を構造化された成分と非構造化された成分の和として分解していたが、この2つの分散成分を区別できないという識別性の問題があるらしい。  
Lerouxモデルでは$\rho$を導入して、潜在効果の分散を構造化分散と非構造化分散の重み付き和として表現してそれに対応していた。  
一方、SørbyeとRueは、BYMモデルでは構造化効果のスケーリングが重要だと指摘していた。そしてRieblerらは、潜在効果を単位分散を持つ非構造化ランダムノイズとスケーリングされた構造化成分の重み付き和に分解するBYM2モデルを提案した。潜在的空間効果のベクトルは、隣接行列から計算される定数に従ってスケーリングされる。このスケーリング処理によって、各構造化成分の分散がほぼ1になり識別可能になる。  
よくわからんが、2つの潜在効果を取り入れると、計算するのが難しくなるからいろいろ頑張ったって話か。
BYM2の潜在効果（ランダムノイズ成分＋空間相関成分）は以下のように表すことができる。

```math
\begin{equation}
\begin{split}

mixture = \sqrt{1‐\rho}\theta+\sqrt{\rho/s}\phi

\end{split}
\end{equation}
```
$\phi$が空間相関成分で、$\theta$がランダムノイズ成分。$\rho$はランダムノイズ成分と空間相関成分のバランスを制御するパラメータ。$\phi$はICARモデルの空間相関変数と同じ定義。$\theta$は、$\theta \sim \mathcal{N}(0,I)$、$s$はスケールファクターで、ICARモデルから生じる一般化分散（$Q=[D-W]$の一般化逆行列）から計算し、$s=\exp[(1/n)\sum_{i=1}^n\ln(diag(Q^-))]$と表す。
以上の定義で説明変数があるBYM2モデルを表すと以下のような式になる。
```math
y_{i}=\sum_{k=1}^{K}x_{i,k}\beta_{k}+\sqrt{1‐\rho}\theta_{i}+\sqrt{\rho/s}\phi_{i}
```
上記を一般化線形モデルに組み込む。

## BYM2モデル構築
モデルの特徴的な部分は一般化分散の逆行列（精度行列）を計算しているところ。$[D-W]$の一般化逆行列`Q_inv`を計算して、`Q_inv`から`scaling_factor`を計算している。単純に逆行列を求めると  `scaling_factor`がNaNになったので`np.linalg.pinv`で一般化逆行列を求めよう。  
後は空間相関成分`ICAR10s`とランダムノイズ成分`theta10s`を定義して、$\sqrt{1‐\rho}\theta+\sqrt{\rho/s}\phi$に従って`mixture`を定義して説明変数と一緒にソフトマックス関数に入れている。
```python
# BYM2モデル構築
with pm.Model() as model_bym2:    
    # coords(次元やインデックスを定義)
    model_bym2.add_coord('data', values=range(X_train.shape[0]), mutable=True)
    model_bym2.add_coord('var', values=explanatory_variables, mutable=True)
    model_bym2.add_coord('obj_var', values=sorted(y_train[objective_variables].unique()), mutable=True)
    
    # 変数
    x = pm.MutableData('x', X_train[explanatory_variables].to_numpy(), dims=('data', 'var'))
    y = pm.MutableData("y", y_train[objective_variables].to_numpy(), dims=('data', ))
    weights = pm.MutableData("weights", sample_ws, dims=('data', ))
    print('x shape', x.eval().shape)
    print('y shape', y.eval().shape)
    print('weights shape', weights.eval().shape)

    # 精度行列の計算
    tau = 1
    N = len(adj_symmetry)
    W = adj_symmetry
    D = np.diag(adj_symmetry.sum(axis=1))  # 対角行列(近傍の数を表す)
    B = np.linalg.inv(D) @ adj_symmetry  # スケーリング隣接行列(行方向の合計が1)
    I = np.eye(len(adj_symmetry))
    #Leroux_precision =  tau2_Dinv @ (rho * W + (1 - rho) * np.eye(len(W)))
    Q = tau*D - tau*adj_symmetry  # 精度行列  # == Q =  tau*D @ I - tau*D @ B
    # add a small jitter along the diagonal
    # Q_perturbed = Q + sparse.diags(np.ones(Q.shape[0])) * max(Q.diagonal()) * np.sqrt(np.finfo(np.float64).eps)
    # Q_perturbed_inv = np.linalg.inv(Q_perturbed)  # (大きな正の数値になり、scaling_factorも巨大な数値になる)
    Q_inv = np.linalg.pinv(Q)  # 擬似逆行列(そのまま逆行列を計算すると大きな負の数値になりscaling_factorはNaNになる)
    # scaling factor
    scaling_factor =  np.exp(np.mean(np.log(np.diag(Q_inv))))
    print('scaling_factor', scaling_factor)
    
    # 推論パラメータの事前分布
    coef_ = pm.Normal('coef', mu=0.0, sigma=1, dims=("var",'obj_var'))  # 各係数の事前分布は正規分布
    intercept_ = pm.Normal('intercept', mu=0.0, sigma=1.0, dims=("obj_var", ))  # 切片の事前分布は正規分布
    print('coef shape', coef_.eval().shape)
    print('intercept shape', intercept_.eval().shape)
    
    # ICARを10個入ったリストを作ってconcatenateしてdata数×class数の空間相関変数としている
    ICAR10s = [pm.ICAR('z_car'+str(i+1), W=adj_symmetry, sigma=1, dims=('data', )).reshape((len(y.eval()), 1)) for i, k in enumerate(classK)]
    ICAR10s = pm.math.concatenate(ICAR10s, axis=1)

    # independent random effect
    theta10s = [pm.Normal("theta"+str(i+1), mu=0, sigma=1, dims=('data', )).reshape((len(y.eval()), 1)) for i, k in enumerate(classK)]
    theta10s = pm.math.concatenate(theta10s, axis=1)

    # 分散
    tau2 = pm.Uniform('tau2', lower=0, upper=100)   # spatial var
    print('tau2 shape', tau2.eval().shape)
    
    # spatial param
    rho = pm.Uniform('rho', lower=0, upper=1, dims=("obj_var", ))
    mixture = pt.sqrt(1 - rho) * theta10s + pt.sqrt(rho / scaling_factor) * ICAR10s
    
    print('ICAR10s type', ICAR10s.type)
    # linear model --> 𝑦 = coef_𝑥 + intercept_ + eps_ + ICARs
    # mu = pm.Deterministic("mu", x.dot(coef_)+intercept_+(mixture*tau2), dims=('data', 'obj_var'))
    print('x.dot(coef_) shape', x.dot(coef_).eval().shape)
    print('ICAR10s*tau type', (ICAR10s*tau).type)
    # psi = pm.Deterministic("psi", pm.math.softmax(mu, axis=1), dims=('data', 'obj_var'))  # axis設定しないとダメ
    psi = pm.Deterministic("psi", pm.math.softmax(x.dot(coef_)+intercept_+(mixture*tau2), axis=1), dims=('data', 'obj_var'))  # axis設定しないとダメ
    #y_pred = pm.Categorical("y_pred", p=psi, observed=y, dims='data')
    y_pred = pm.Potential('y_pred', (weights * pm.logp(pm.Categorical.dist(p=psi), y)).sum(axis=0), dims=('data', ))
    
# モデル構造
modeldag = pm.model_to_graphviz(model_bym2)
display(modeldag)
```
![image-11.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/9135dead-4626-11ab-36cf-7991fabd7c96.png)

```python
%%time
# MCMC実行
# nutpieで実行
with model_bym2:
    # MCMCによる推論
    trace = pm.sample(draws=3000, tune=1000, chains=3, nuts_sampler="nutpie", cores=3, random_seed=1, return_inferencedata=True)#, idata_kwargs={"log_likelihood": False})

# 保存
model_dir = '/content/drive/MyDrive/satelite/model'
os.makedirs(model_dir, exist_ok=True)
# データの保存 to_netcdfの利用
trace.to_netcdf(os.path.join(model_dir, 'model_bym2.nc'))

# Load
model_dir = '/content/drive/MyDrive/satelite/model'
os.makedirs(model_dir, exist_ok=True)
# データの読み込み from_netcdfの利用
trace = az.from_netcdf(os.path.join(model_dir, 'model_bym2.nc'))
```

trace plotで収束を確認。  
クラスによって$\rho$パラメータが結構違う ！
収束もしてそう。
```python
# plot_trace
az.plot_trace(trace, backend_kwargs={"constrained_layout":True}, var_names=["coef","intercept","tau2","rho"])#+['z_car'+str(i) for i in range(1,11)])
plt.show()
```
![image-12.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e706c15e-97e2-2f86-48b0-ea4c131a0d6e.png)


$\hat{R}$でも収束の確認。  
問題なさそう。
```python
# MCMCの収束を評価
rhat_vals = az.rhat(trace).values()

# 最大のRhatを確認
result = np.max([np.max(i.values) for i in rhat_vals])# if i.name in ["coef","intercept",'z_car','mu','theta','tau']])
print('Max rhat:', result)
# 1.1以上のRhatを確認
for i in rhat_vals:
    if np.max(i.values)>=1.1:
        print(i.name, np.max(i.values), np.mean(i.values), i.values.shape, sep='  ====>  ')
# >> Max rhat: 1.0102289229229913
```

各クラスごとの空間相関パラメータ$\rho$を確認。 
水田やその他の農用地などは空間相関成分の影響が大きいが、荒地や鉄道、道路はランダムノイズ成分の方が影響が大きい。  
鉄道や道路などは面積も小さいので空間相関が小さいというのは直感的にも合う結果。
```python
# クラスごとの空間相関
spatial_corr = {area_use_categories_le[c]:round(r,2) for c, r in enumerate(trace['posterior']['rho'].mean(dim=["chain", "draw"]).values)}
display(spatial_corr)
```
![image-23.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6f89ec0a-a9cd-753d-f08a-3c583b90297e.png)

学習データの精度確認。  
悪くない。
```python
# 各クラスの事後確率theta
softmax_result = pd.DataFrame(trace['posterior']['psi'].mean(dim=["chain", "draw"]).values)
y_pred = softmax_result.idxmax(axis=1)
print(sklearn.metrics.classification_report(y_train[objective_variables].to_numpy(), y_pred.ravel()))
cm = sklearn.metrics.confusion_matrix(y_train[objective_variables].to_numpy(), y_pred.ravel())
print(cm)
```
![image-14.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/cbe800ed-2f07-7764-6660-c878bd8cdccf.png)

テストデータの精度確認。
```python
# 学習データ、テストデータの座標を取得
train_coord = pd.DataFrame({'x':X_train['geometry'].x.to_list(), 'y':X_train['geometry'].y.to_list()})
test_coord = pd.DataFrame({'x':X_test['geometry'].x.to_list(), 'y':X_test['geometry'].y.to_list()})
# 学習データでknn作成
knn = sklearn.neighbors.NearestNeighbors(n_neighbors=5)
knn.fit(train_coord)
# テストデータの各サンプルと最も近い学習データのindexを取得
distances, indices = knn.kneighbors(test_coord)
test_coord['indices'] = indices[:,0]

# テストデータと最も近い学習データの空間相関変数を取得する
z_cars_arr = np.concatenate([trace['posterior']['z_car'+str(i+1)].mean(dim=["chain", "draw"]).values.reshape(-1,1) for i, k in enumerate(classK)], axis=1)  # 学習データの空間相関変数の計算
z_cars_arr_test = z_cars_arr[list(indices[:,0]),:]  # テストデータと最も近い学習データの空間相関を各テストデータに対して計算
theta_arr = np.concatenate([trace['posterior']['theta'+str(i+1)].mean(dim=["chain", "draw"]).values.reshape(-1,1) for i, k in enumerate(classK)], axis=1)  # 学習データの空間相関変数の計算
theta_arr_test = theta_arr[list(indices[:,0]),:]  # テストデータと最も近い学習データの空間相関を各テストデータに対して計算
tau_arr = trace['posterior']['tau2'].mean(dim=["chain", "draw"]).values
rho_arr = trace['posterior']['rho'].mean(dim=["chain", "draw"]).values
mixture = (np.sqrt(1 - rho_arr) * theta_arr + np.sqrt(rho_arr / scaling_factor) * z_cars_arr) * tau_arr  # 分散をかける
mixture_test = (np.sqrt(1 - rho_arr) * theta_arr_test + np.sqrt(rho_arr / scaling_factor) * z_cars_arr_test) * tau_arr  # 分散をかける
print(mixture.shape)
print(mixture_test.shape)
# >> (851, 10)
# >> (9794, 10)

# 未知データへの適用
# 回帰係数の推定値
coefs = trace['posterior']['coef'].mean(dim=["chain", "draw"]).values
# 切片の推定値
intercepts = trace['posterior']['intercept'].mean(dim=["chain", "draw"]).values
# 線形モデル式
mu = X_test[explanatory_variables].to_numpy().dot(coefs) + intercepts + mixture_test  # 回帰式（係数と切片）と空間相関変数
# ソフトマックス関数に入れる
m = softmax(mu, axis=1)  # ソフトマックス関数に入れて一般化線形モデルへ
# 確率が最大のクラスを取得
m_class = m.argmax(axis=1)  # 最も確率が高いクラスを所属クラスとする
# 精度計算
print(sklearn.metrics.classification_report(y_test[objective_variables].to_numpy(), m_class.ravel()))
cm = sklearn.metrics.confusion_matrix(y_test[objective_variables].to_numpy(), m_class.ravel())
print(cm)
```
[前回の記事](https://qiita.com/drafts/886d35561a4f23653dc4/edit)のICARと同程度。
![image-15.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/a4a1acb2-f814-f502-a647-4a0071169c98.png)

分類結果確認。
```python
# 可視化
# 学習データ
results = y_train.copy()
results['pred'] = y_pred.ravel()
results['diff'] = results[objective_variables]-results['pred']
results.loc[(results['diff']==0), 'diff'] = 0
results.loc[(results['diff']!=0), 'diff'] = 1
point_plot(results, re_shape_tsukuba_mirai_2RasterCrs, linewidth=0.05, markersize=5, title='土壌分類', vmin=0, vmax=1)

# テストデータ
results = y_test.copy()
results['pred'] = m_class.ravel()
results['diff'] = results[objective_variables]-results['pred']
results.loc[(results['diff']==0), 'diff'] = 0
results.loc[(results['diff']!=0), 'diff'] = 1
point_plot(results, re_shape_tsukuba_mirai_2RasterCrs, linewidth=0.05, markersize=1, title='土壌分類', vmin=0, vmax=1)
```
学習データ
![image-16.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/15f6081f-797c-71d2-d518-74c1db70715d.png)
テストデータ
![image-17.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/dd6645e4-ddf2-81b9-af76-64d1db381ced.png)

空間パターンを見るため、空間相関変数を可視化。
```python
# 学習データの空間相関変数を可視化
spatial_pattern(ground_truth_2RasterCrs_concat_crop_exploded_stats_2point
                , mixture
                , X_train
                , classK
                , re_shape_tsukuba_mirai_2RasterCrs
                , area_use_categories_le
                , node_name='z_car', figsize=(8,6), subplot_num=(3,4), markersize=3)
# テストデータの空間相関変数（最も近い学習データの空間相関変数を取得した結果）を可視化
spatial_pattern(ground_truth_2RasterCrs_concat_crop_exploded_stats_2point
                , mixture_test
                , X_test
                , classK
                , re_shape_tsukuba_mirai_2RasterCrs
                , area_use_categories_le
                , node_name='z_car', figsize=(8,6), subplot_num=(3,4), markersize=1)
```
学習データ
![image-18.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/b2f6edff-fdcb-1459-d1b9-2607d209262e.png)
テストデータ
![image-19.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/db2fe14f-1b62-01d6-c9ae-8aebde7529ec.png)

水田やその他の農用地など空間相関が大きいクラスは、そのクラスの地域から離れると滑らかに空間相関変数が減衰するようなパターンが可視化されている。一方、空間相関が小さい荒地や道路、鉄道は実際にそのクラスに分類された地域から離れたらすぐに空間相関変数が0に近い値になっており、滑らかに空間相関変数が小さくなるような傾向は見られなく空間相関が小さいことがわかる。これは直感にも合うと思われる。

$\rho$を固定して、$\rho=0.5$と$\rho=0$の2パターンの結果を見てみると以下のようになった。  
- $\rho=0.5$
![image-20.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d702330d-642c-0e77-d8bb-9eca4e412ee2.png)
- $\rho=0$
![image-21.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/1c2294d8-de1a-53ea-3013-9d0aa83beee5.png)

$\rho$が小さくなると、すべてのクラスで空間相関が小さい時のパターンが見える。モデルの中で$\rho$の値は効いていることがわかる。  

うん、BYM2モデルはいい感じ！

# おわりに
今回はCARモデルの中の、LerouxモデルとBYM2モデルの構築を行った。  
Lerouxモデルはどこかミスがあるかもしれない…。空間相関のパラメータが0ってことはあるまい…。  
BYM2モデルはPyMC公式のExample：[「The Besag-York-Mollie Model for Spatial Data」](https://www.pymc.io/projects/examples/en/latest/spatial/nyc_bym.html)で詳しく解説されていて、コードも載っていたので良い感じにモデルができた。土壌クラスによっては空間相関が小さいものがあってそれが鉄道や道路など空間相関が小さいことに違和感が無いクラスだった。逆に広く分布している水田などは空間相関が大きく推定されていてこれも違和感が無い。  
CARモデルはとりあえずBYM2モデルで良いかもしれない。ICARやLerouxを使った方が良いケースとかあるのかな…。Lerouxモデルはもうちょっとうまくできないか考えたいなぁ。

以上！