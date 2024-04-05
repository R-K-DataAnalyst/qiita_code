# はじめに
pythonのmatplotlibで日本語を扱うときに豆腐(→ □□)になることがあるだろう。これを避ける方法を調べていると、"サイトから「ipaexg.ttf」をダウンロードして～～"って説明が多いと思う。なんかめんどうだなーって思ったことがある。そもそもすでに標準に持っているフォントじゃダメなの？って思ってたらできるやん。

# コード
以下のようにすると豆腐になる。

```python
import seaborn as sns
import numpy as np
sns.set()
fig=plt.figure()
ax=plt.subplot(1,1,1)
ax.plot(np.arange(-10,11),np.arange(-10,11)**2)
ax.set_title('二次曲線')
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/023e507d-1c79-92cb-4813-7b80899d1a02.png)

しかし以下のmsgothic.ttcを使えば、日本語を可視化できる。（おそらくWindowsならデフォルトで入っているはず）
matplotlib.font_managerのfindSystemFonts()で使用できるフォント一覧が取得できるので、その中で"msgothic"と書かれているものだけを取り出して、そのフォントをplt.rcParams['font.family']で設定してあげると、日本語化できる。

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sns.set()
jpn_fonts=list(np.sort([ttf for ttf in fm.findSystemFonts() if 'ipaexg' in ttf or 'msgothic' in ttf or 'japan' in ttf or 'ipafont' in ttf]))
jpn_font=jpn_fonts[0]
prop = fm.FontProperties(fname=jpn_font)
print(jpn_font)

plt.rcParams['font.family'] = prop.get_name()
fig=plt.figure()
ax=plt.subplot(1,1,1)
ax.plot(np.arange(-10,11),np.arange(-10,11)**2)
ax.set_title('二次曲線')
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/ef2761a4-77ac-7d5c-7458-f5738ed283bc.png)

# おわりに
定常的にmsgothicを使いたいなら、matplotlibrcを書き換えたりしないといけないけど、突貫で豆腐を避けるならmsgothicをfindSystemFonts()で取り出してplt.rcParams['font.family']で定義してあげればいいだけなので"サイトから「ipaexg.ttf」をダウンロードして～～"って作業はいらないと思われる。MacやLinuxの場合は知らない。
以上！

※追記
あ、matplotlibは昔ttcファイルに対応していなかったからipaexg.ttfをダウンロードしてたんだね…知らなかった…。

※追記2
WSL2(Windows Subsystem for Linux 2)上でpythonを使っていて、豆腐を避けたい人はデフォルトで日本語対応のフォントは入っていないのでインストールする必要がある。
WSL(Ubuntuを使用)上のターミナルで以下のコマンドでipafontをインストール。

```{}
sudo apt install -y fonts-ipafont
```
すると、日本語対応のフォントがいくつか入るのでどれかを選んで使用すればOK。

```{python}
import matplotlib.font_manager as font_manager
import numpy as np
jpn_fonts=list(np.sort([ttf for ttf in font_manager.findSystemFonts() if 'japan' in ttf or 'ipafont' in ttf]))
jpn_font=jpn_fonts[0]
prop = font_manager.FontProperties(fname=jpn_font)

display(jpn_fonts)
print(jpn_font)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/d1028c96-8341-3aff-67ac-3f2d9bf2e0c5.png)
