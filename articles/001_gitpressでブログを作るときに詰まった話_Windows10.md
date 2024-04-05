# gitpressでブログを作るときに詰まった話（Windows10）

## gitpressって何なの？
GitHubのリポジトリを誰でも簡単に個人ブログへと変換できる無料のWebサービス。
[gitpress](https://gitpress.io/)

やり方は以下のサイトを参考にしました。
[Pushするだけ！GitHubのリポジトリを個人ブログに変えてくれる【GitPress】を使ってみた！](https://paiza.hatenablog.com/entry/2019/06/19/Push%E3%81%99%E3%82%8B%E3%81%A0%E3%81%91%EF%BC%81GitHub%E3%81%AE%E3%83%AA%E3%83%9D%E3%82%B8%E3%83%88%E3%83%AA%E3%82%92%E5%80%8B%E4%BA%BA%E3%83%96%E3%83%AD%E3%82%B0%E3%81%AB%E5%A4%89%E3%81%88%E3%81%A6 "GitPress")

## Windowsでの環境構築
・githubアカウントを作成
　[github](https://github.com/)
・GitPressが公式に用意しているテンプレート(ボイラープレート)をフォーク
・GitHubアカウントと連携
・ブログにしたいリポジトリを選択

といった流れです。
参考にした上記サイトではgithub上から新しいファイルを作り、markdownで記事を書いていました。
でもローカルPC上で記事を書いてgithubにpushして記事をアップロードしたいときもありますよね？（無いかな笑）
そんなときはどうするんだろうか。github初心者の私には荷が重いか？？やってみよう。

## ローカルPC上で記事を書いてgithubのリモートリポジトリにpush
まず以下の2つを行います。
・Windows用のGitをinstall
　[Gitダウンロードページ](https://gitforwindows.org/)
・Git GUIで任意のフォルダーにリポジトリ作成
その後、test.mdとして適当にmarkdownで記事を書きます。

```
# こんにちは初投稿です
## これはテストです
あはははは
```

これをgit bashを開いて、githubにpushします。(コマンドプロンプトからでもいけた)

```
#初期化
git init
#コンテンツ（ファイルなど）を見つけてインデックスに追加
git add .
#コミット
git commit -m ‘test’
#リモートリポジトリの登録
git remote add origin "記事を上げるgithubリポジトリのURL(https://github.com/(アカウント名)/boilerplate.git)"
#プッシュ
git push -u origin master
```
```
! [rejected]        master -> master (fetch first)
```


あれ？怒られた？github初心者の私はここでけっこう詰まってしまいました。
いろいろ調べるとこんな記事が。
[git push -u origin masterでrejectにハマる](https://qiita.com/watsuyo_2/items/aa95e54c4974a80123e9)
[git pushがrejectされたときの対応方法](https://www.softel.co.jp/blogs/tech/archives/3569)

この記事のコメントに
>README.mdとか、リモートリポジトリにあって、ちゃんとマージできてなかったから

と書かれていたので、

```
#リモートの変更を取ってきて
git fetch
#マージする
git rebase origin/master
```

これをやった後に、再度以下を実行

```
git init
#コンテンツ（ファイルなど）を見つけてインデックスに追加
git add .
#コミット
git commit -m ‘test’
#プッシュ
git push -u origin master
```
無事以下のようにtest.mdがpushされました。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/dadbff28-bd4b-6374-4324-ef74c6e50bdd.png)

さて、gitpressの自分のページにいってみると...
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/f938ebbe-a23d-79c8-41f1-47ad98afb984.png)

記事ができている！これでブログをはじめられるぜ！
無事ブログをはじめられそうです。（ちゃんと書くかは不明）

## まとめ
gitpressでブログを始めるための流れと、記事をpushする際に詰まった時の対処を書きました。
「技術系の話を書くときはQiitaで良くない？ブログいる？」という話にはお答えできませんのであしからず...。
（業界に関するポエムを書く時とかはブログがいいかも...）
