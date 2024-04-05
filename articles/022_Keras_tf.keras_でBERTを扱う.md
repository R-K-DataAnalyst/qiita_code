# はじめに
世間はPytorch一択なのだろうか。私は信じたい、Keras(Tensorflow)の力を。

ということでKerasでBERTモデルを扱ったので備忘録を残す。

いろいろ調べていると、huggingfaceのtransformersライブラリを使うときは、Pytorchを使ってる人が多いと感じた。自分も[過去の記事](https://qiita.com/chicken_data_analyst/items/15c0046062c6e016f467)ではPytorchを使っていた。ただ慣れているのはKerasなので、Kerasでtransformersのモデルを扱えるようになっておきたいなと思い、お勉強したという経緯。
先人たちのいろいろな知見をつまみ食いしながらお勉強したので、自分用にこの記事を書いて知見を1つにまとめたい。

今回書くのは以下2つ。
- Kerasを使ったMaskedLMタスクによるBERT学習方法
- Kerasを使ったBERTのファインチューニング（文書分類）の方法

# 参考
- [[huggingface] masked_language_modeling](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling)
- [[Keras公式] Pretraining BERT with Hugging Face Transformers](https://keras.io/examples/nlp/pretraining_BERT/)
- [[Github] transformers/examples/pytorch/language-modeling
/run_mlm.py](https://github.com/huggingface/transformers/blob/v4.18.0/examples/pytorch/language-modeling/run_mlm.py)
- [『Transformerによる自然言語処理』のRoBERTa事前訓練のコードを、データをhuggingface/datasetsで読み込むように書き直す](https://nikkie-ftnext.hatenablog.com/entry/replace-linebylinetextdataset-datasets-library)
- [[huggingface] Main classes：to_tf_dataset()](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_tf_dataset)
- [[Github] TFBertModel.predict() and TFBertModel() return different results. #16160](https://github.com/huggingface/transformers/issues/16160)
- [[stackoverflow] batch_size in tf model.fit() vs. batch_size in tf.data.Dataset](https://stackoverflow.com/questions/62670041/batch-size-in-tf-model-fit-vs-batch-size-in-tf-data-dataset)
- [huggingface Tokenizer の tokenize, encode, encode_plus などの違い](https://zenn.dev/hellorusk/articles/7fd588cae5b173)
- [kaggleでよく使う交差検証テンプレ(Keras向け)](https://zenn.dev/monda/articles/kaggle-cv-template)
- [TensorFlow(Keras)とBERTで文章分類モデルを作る方法](https://htomblog.com/python-tensorflowbert)
- [[Github] Transfer learning & fine-tuning](https://github.com/keras-team/keras-io/blob/master/guides/ipynb/transfer_learning.ipynb)
- [[Github] Difference between CLS hidden state and pooled_output #7540](https://github.com/huggingface/transformers/issues/7540)
- [TensorFlowで使えるデータセット機能が強かった話](https://qiita.com/Suguru_Toyohara/items/820b0dad955ecd91c7f3)

# 結論
さらっと結論を書く。
- transformersライブラリのクラスは、```TFAutoModel```や```TFBertForMaskedLM```のようにTFが付いたクラスを使う
- MaskedLMを実施する時、DatasetDictクラスのデータに対して、```to_tf_dataset()```を適用するとKeras(Tensorflow)で扱えるデータセットになる
- ファインチューニングする時は、```TFAutoModel.from_pretrained()```で事前学習済みのBERTモデルを読み込んで、[CLS]トークンの最終隠れ層```<BERT MODEL OUTPUT>.last_hidden_state[:,0,:]```を後続のレイヤーにつなぐ

# Kerasを使ったMaskedLMタスクによるBERT学習方法
KerasでMaskedLMタスクによる学習を実施する方法を書く。

まず必要なライブラリをimport。（TPUを使ったので、TPUを使う設定も入れている。）
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import scipy
import gc
import pickle
import os
import collections
import random
import string
import re
import sklearn

import tensorflow as tf  # 2.13.0

import transformers  # 4.35.2
from transformers import AutoTokenizer, TFAutoModel, TFBertForMaskedLM, TFBertForPreTraining
from transformers import BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import AdamWeightDecay

from datasets import load_dataset
print(tf.__version__)
print(transformers.__version__)

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
```

トークン長、マスクする確率、事前学習済みモデルの名前、テキストデータのカラム名の定義。[Keras公式](https://keras.io/examples/nlp/pretraining_BERT/)のページを参照している。
```python
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding
MLM_PROB = 0.2  # Probability with which tokens are masked in MLM
MODEL_CHECKPOINT = "bert-base-uncased"  # Name of pretrained model from 🤗 Model Hub
text_column_name = 'text'
```

データの読み込み。Wikiのデータを読み込む。DatasetDictクラスとして読み込まれる。
```python
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
raw_datasets
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/c3682204-d207-a54f-10b3-8a6bc1d22c3b.png)

ちなみに自前のデータをDatasetDictクラスとして読み込む場合、前処理などした後1度ファイルとして出力してからload_datasetで読み込めばいい。
```python
#### 例 ####
# 前処理したデータ
data['text'].to_csv('tmp.txt', index=False, header=False)
data_valid['text'].to_csv('tmp_valid.txt', index=False, header=False)

dataset_files = {
    "train": ["tmp.txt"],
    "validation": ["tmp_valid.txt"],
}
raw_datasets = load_dataset(text_column_name, data_files=dataset_files)
```

次にtokenizerとData Collatorを定義。```DataCollatorForLanguageModeling```によってテキストのマスク処理の内容を定義できる。
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB, return_tensors="tf")
```

次に、文書をトークンナイズして、MaskedLMタスクで学習させるためのdatasetに加工する。
以前は```LineByLineTextDataset```を使っていたが、今は非推奨らしく（[『Transformerによる自然言語処理』のRoBERTa事前訓練のコードを、データをhuggingface/datasetsで読み込むように書き直す](https://nikkie-ftnext.hatenablog.com/entry/replace-linebylinetextdataset-datasets-library)）、datasetsライブラリを使う必要がある。
[transformersのGithub](https://github.com/huggingface/transformers/blob/v4.18.0/examples/pytorch/language-modeling/run_mlm.py)から、```tokenize_function```という関数を拝借してmapを適用すれば、それが可能になる。
```python
def tokenize_function(examples):
    # Remove empty lines
    examples[text_column_name] = [
        line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples[text_column_name],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,  # https://discuss.huggingface.co/t/why-use-batched-true-in-map-function/18042
    num_proc=None,
    remove_columns=[text_column_name],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset line_by_line",
)
tokenized_datasets
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/c8838086-4817-f56e-1281-8e7c7002259d.png)


と、ここまでいろいろ処理してきたが、ここまでの処理はPytorchでもKeras(Tensorflow)でも変わらない。

次は、ここまで処理してきたデータをKeras(Tensorflow)用に加工する必要がある。そのために```to_tf_dataset()```関数を使う。（[[huggingface] Main classes：to_tf_dataset()](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_tf_dataset)）
```to_tf_dataset()```関数の中の引数collate_fnで先ほど定義したdata_collatorを指定すると、マスク処理されたデータにも加工される。
ここでbatch_sizeも指定しているので、モデルを学習させるときの引数でバッチサイズを指定する必要はなくなる。（[[stackoverflow] batch_size in tf model.fit() vs. batch_size in tf.data.Dataset](https://stackoverflow.com/questions/62670041/batch-size-in-tf-model-fit-vs-batch-size-in-tf-data-dataset)）
```python
np.object = object  # エラーが出るので仕方なし。非推奨。npのバージョン<1.24にすればいいらしい。(https://stackoverflow.com/questions/75069062/module-numpy-has-no-attribute-object)
ds_train = tokenized_datasets["train"].to_tf_dataset(
   columns=['input_ids', 'token_type_ids', 'attention_mask'],
   label_cols=["labels"],
   shuffle=True,
   batch_size=64,
   collate_fn=data_collator,
)
ds_valid = tokenized_datasets["validation"].to_tf_dataset(
   columns=['input_ids', 'token_type_ids', 'attention_mask'],
   label_cols=["labels"],
   shuffle=True,
   batch_size=64,
   collate_fn=data_collator,
)
ds_train
```

これでTensorflowでも扱えるデータに変わったので、あとは学習させるだけ。
Pytorchの時は、```BertForMaskedLM```でモデルを定義していたが、Keras(Tensorflow)の時は、```TFBertForMaskedLM```で定義する。
```python
%%time
# TPU用の設定
# バッチサイズ
batch_size_per_replica = 64

# データセットのサンプル数
total_samples = len(tokenized_datasets["train"])
total_samples_v = len(tokenized_datasets["validation"])

# steps_per_epoch の計算
steps_per_epoch = total_samples // batch_size_per_replica
val_steps_per_epoch = total_samples_v // batch_size_per_replica
print(steps_per_epoch, val_steps_per_epoch)
# > 371 38

with tpu_strategy.scope():
    config = BertConfig()
    model = TFBertForMaskedLM(config)
    # オプティマイザの設定
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    #optimizer = AdamWeightDecay(learning_rate=1e-3, weight_decay_rate=0.0)
    model.compile(optimizer=optimizer)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# batch_sizeの指定はいらない
model.fit(x=ds_train, validation_data=ds_valid, epochs=100
          , callbacks=[callback]
          , steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch)
# > Wall time: 2h 10min 57s
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/8f560906-d854-f9ae-7e9d-0ac17c653b30.png)

ちゃんと学習できているか確認する。例文の単語を1つ[MASK]として隠して、それを当てられるか検証する。今回は学習データの中から例文を取る。以下の例文の”historic”の部分を[MASK]に変える。
```
galveston is home to six historic districts with over 60 structures listed representing architectural significance
　　　　　　　　　　　　　　↓
galveston is home to six [MASK] districts with over 60 structures listed representing architectural significance
```
実際に推論してみる。（[[huggingface] masked_language_modeling](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling)）
ちゃんと"historic"を推論できた。
```python
mlm_bert = model  # モデルの変数名変えているだけ

# 例文
tx = 'galveston is home to six [MASK] districts with over 60 structures listed representing architectural significance'
# トークンナイズ
inputs = tokenizer(tx, return_tensors="tf")
logits = mlm_bert(**inputs).logits
mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)
predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
tokenizer.decode(predicted_token_id)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/bd50c471-7abd-5bf1-e263-67b4d0d0bcae.png)

ちなみに、全く学習させていないモデルで推論してみると不正解だったのでちゃんと学習できていたようだ。
```python
config = BertConfig()
model_tmp = TFBertForMaskedLM(config)
logits = model_tmp(**inputs).logits
mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)
predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
tokenizer.decode(predicted_token_id)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e43b35cf-9807-e813-8813-24cf569bf699.png)

# Kerasを使ったBERTのファインチューニング（文書分類）の方法
Kerasでファインチューニングする方法を書く。

まず必要なライブラリをimport。（TPUではなく、GPUを使った。）
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import scipy
import gc
import pickle
import os
import collections
import random
import string
import re
import sklearn
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf  # 2.13.0
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

import transformers  # 4.35.2
from transformers import AutoTokenizer, TFAutoModel, TFBertForMaskedLM, TFBertForPreTraining
from transformers import BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import AdamWeightDecay

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from datasets import load_dataset
print(tf.__version__)
print(transformers.__version__)
# > 2.13.0
# > 4.35.0
```
データは"20 newsgroups text dataset"を使う。記号とか数字とかの前処理を適用して、TrainとValidationデータを分けておく。
```python
# 小文字化、記号・丸囲い数字除去、全角を半角など前処理関数
def preprocessing(text):
    text = text.lower()  # 小文字化
    text = re.sub('\r\n','',text)  # \r\nをdelete
    text = re.sub('\n',' ',text)  # \r\nをdelete
    text = re.sub(r'\d+','0',text)  # 数字列をdelete
    ZEN = "".join(chr(0xff01 + i) for i in range(94)).replace('－','').replace('．','').replace('！','').replace('？','')  # 全角文字列一覧
    HAN = "".join(chr(0x21 + i) for i in range(94)).replace('-','').replace('.','').replace('!','').replace('?','')  # 半角文字列一覧
    ALL=re.sub(r'[a-zA-Zａ-ｚＡ-Ｚ\d]+','',ZEN+HAN).replace('.','').replace('．','').replace('!','').replace('！','').replace('?','').replace('？','')  # ドットは残す
    code_regex = re.compile('['+'〜'+'、'+'。'+'~'+'*'+'＊'+ALL+'「'+'」'+']')
    text = code_regex.sub('', text)  # 記号を消す

    ZEN = "".join(chr(0xff01 + i) for i in range(94))
    HAN = "".join(chr(0x21 + i) for i in range(94))
    ZEN2HAN = str.maketrans(ZEN, HAN)
    HAN2ZEN = str.maketrans(HAN, ZEN)
    text=text.translate(ZEN2HAN)  # 全角を半角に
    text = text.replace('-','')
    text = re.sub('[ 　]+', ' ', text)
    text = re.sub('[..]+', '.', text)
    text = re.sub('[\t]+', '', text)
    return text

text_column_name = 'text'
data = datasets.fetch_20newsgroups()
data_valid = pd.DataFrame({text_column_name:data.data, 'label':data.target}).iloc[10000:,:].copy()
data = pd.DataFrame({text_column_name:data.data, 'label':data.target}).iloc[0:10000,:]
data[text_column_name] = data[text_column_name].map(lambda x: preprocessing(x))
data_valid[text_column_name] = data_valid[text_column_name].map(lambda x: preprocessing(x))
print(len(data),len(data_valid))
display(data.head())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/1cb52bf5-43cf-abfb-3b7e-cbc88adea915.png)

ストップワードの除去。
```python
spacy_model = spacy.load('en_core_web_sm')
stopwords_spacy = list(spacy_model.Defaults.stop_words)
stop_words_nltk = (stopwords.words('english'))
stopWords = list(set(stop_words_nltk+stopwords_spacy))
stopWordsDict = {i:'' for i in stopWords+['']}
# tokenized_corpus_without_stopwords = [i for i in tokenized_corpus_nltk if not i in stop_words_nltk]
# print('Tokenized corpus without stopwords:', tokenized_corpus_without_stopwords)
def transStopWords(text):
    word_list_noun = [w for w in (text.split(' ')) if w not in stopWords+['']]
    return ' '.join(word_list_noun)

data[text_column_name] = data[text_column_name].map(lambda x: transStopWords(x))
data_valid[text_column_name] = data_valid[text_column_name].map(lambda x: transStopWords(x))
display(data.head())
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/15122db9-e56b-7d31-8696-089821bd17b9.png)

分類ラベルをOne-Hot表現に変えておく必要があるので変換処理を実施。
```python
num_class = data['label'].unique().shape[0]
data_label = tf.keras.utils.to_categorical(data['label'], num_classes=num_class)
data_valid_label = tf.keras.utils.to_categorical(data_valid['label'], num_classes=num_class)
```

tokenizerの読み込み。今回のファインチューニングではDeBERTaを使うので、DeBERTaのtokenizerをload。
```python
max_length = 200
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
```

データをまとめて、トークンナイズ処理する。複数の文書のテキストをトークンナイズする場合は、```batch_encode_plus()```を使う。（[huggingface Tokenizer の tokenize, encode, encode_plus などの違い](https://zenn.dev/hellorusk/articles/7fd588cae5b173)）

```python
train_tokens = tokenizer.batch_encode_plus(
    data[text_column_name].to_list(),
    padding = "max_length",
    max_length = max_length,
    truncation = True, return_tensors='tf', add_special_tokens=True
)

valid_tokens = tokenizer.batch_encode_plus(
    data_valid[text_column_name].to_list(),
    padding = "max_length",
    max_length = max_length,
    truncation = True, return_tensors='tf', add_special_tokens=True
)
```

次にモデルを構築する。
構築方法は「[TensorFlow(Keras)とBERTで文章分類モデルを作る方法](https://htomblog.com/python-tensorflowbert)」を参考にした。

```TFAutoModel.from_pretrained('microsoft/deberta-v3-base')```でDeBERTaの事前学習済みモデルをloadしている。（Pytorchの場合```AutoModel.from_pretrained```を使う。）

「[TensorFlow(Keras)とBERTで文章分類モデルを作る方法](https://htomblog.com/python-tensorflowbert)」では、pooler_outputの出力を使っているが、DeBERTaはpooler_outputの出力を返さないので、last_hidden_stateを取り出した。
pooler_outputとlast_hidden_stateの違いは「[[Github] Difference between CLS hidden state and pooled_output #7540](https://github.com/huggingface/transformers/issues/7540)」に書いている。

DeBERTaの事前学習済みの層の学習はしないように、```bert_model.trainable = False```にしている。
最初、```bert_model.trainable = True```で実行したのだが、epochが進んでもval_accが0.5に張り付いたまま全く向上しない事象が起き、色々調べていると「[[Github] Transfer learning & fine-tuning](https://github.com/keras-team/keras-io/blob/master/guides/ipynb/transfer_learning.ipynb)」を見つけた。
「[[Github] Transfer learning & fine-tuning](https://github.com/keras-team/keras-io/blob/master/guides/ipynb/transfer_learning.ipynb)」によると、
>訳：ランダムに初期化された学習可能な層と、事前に学習された特徴を保持する学習可能な層が混在している場合、ランダムに初期化された層は学習中に非常に大きな勾配更新を引き起こし、事前に学習された特徴を破壊してしまいます。

ということで、事前学習の特徴を破壊してしまうので、BERTの層はFreezeさせてファインチューニングした方がいいらしい。
BERTの層を再調整したい場合は、1度ファインチューニングした後、低い学習率で```bert_model.trainable = True```にしてもう1度学習を回した方がいいとのこと。
>訳：モデルが新しいデータに収束したら、ベースモデルの全部または一部のフリーズを解除して、非常に低い学習率でモデル全体をend-to-endで再学習させることができます。

```bert_model.trainable = False```に変更して学習させると、epochが進むと同時にval_accも向上し始めたので、```bert_model.trainable = True```で学習したときは事前学習済みの重みに悪影響があったっぽい。

今回は時間もかかるので、再度```bert_model.trainable = True```にしてDeBERTaの事前学習済みの層を再調整することはしないが、覚えておきたい。
```python
def get_model(max_length, num_classes):
    # input_idsを受け取る層
    input_ids = L.Input(
        shape = (max_length), dtype = tf.int32, name = "input_ids"
    )

    # attention_maskを受け取る層
    attention_mask = L.Input(
        shape = (max_length), dtype = tf.int32, name = "attention_mask"
    )

    # token_type_idsを受け取る層
    token_type_ids = L.Input(
        shape = (max_length), dtype = tf.int32, name = "token_type_ids"
    )
    
    # BERTモデル
    bert_model = TFAutoModel.from_pretrained('microsoft/deberta-v3-base')
    bert_model.trainable = False  # 最初はFalseで実施
    transformer_outputs = bert_model(
        {"input_ids": input_ids,
         "attention_mask": attention_mask,
         "token_type_ids": token_type_ids}
    )
    pooler_output = transformer_outputs.last_hidden_state[:,0,:]  # 最終隠れ層
    #pooler_output = transformer_outputs.pooler_output

    # BERTの出力->クラス数に変換する層
    outputs = L.Dense(units = num_classes, activation = "softmax")(pooler_output)  # 'sigmoid'

    # 定義した層からモデルを作成
    model = M.Model(
        inputs = [input_ids, attention_mask, token_type_ids],
        outputs = outputs
    )

    # 最適化手法と損失関数を定義
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
        loss = 'categorical_crossentropy',  # "binary_crossentropy",
        #metrics=[tf.keras.metrics.AUC()]
        metrics=['acc']
    )
    return model

model = get_model(max_length, num_class)
model.summary()
```
（事前学習済みの層は学習させない）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/e168c13c-b9ca-b1bd-6ff3-bb93687954c6.png)

モデルの構築が済んだので、学習させていく。
まずトークンナイズされたデータをTensorflowのDataset APIでデータセット化する。
```batch_encode_plus()```でトークンナイズされたデータ(train_tokens)に入っている"input_ids"、"attention_mask"、"token_type_ids"と、One-Hot表現の正解ラベルのデータ(data_label)を```tf.data.Dataset.from_tensor_slices()```に渡せばOK。
データセットに対して```.batch(8)```のように書いてあげると、バッチ化できるのでモデルの```model.fit()```時にバッチサイズを指定してあげる必要はなくなる。
あとは、普段のように```ModelCheckpoint```や```EarlyStopping```を指定して、学習させてあげればOK。（交差検証のコードは「[kaggleでよく使う交差検証テンプレ(Keras向け)](https://zenn.dev/monda/articles/kaggle-cv-template)」を参照した。）

```python
%%time
valid_scores = []
models = []
skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
for fold, (train_indices, valid_indices) in enumerate(skf.split(train_tokens['input_ids'].numpy(), data.label.to_numpy())):
    # データセットをTensorflo用に加工
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": train_tokens["input_ids"].numpy()[train_indices],
         "attention_mask": train_tokens["attention_mask"].numpy()[train_indices],
         "token_type_ids": train_tokens["token_type_ids"].numpy()[train_indices]},
        data_label[train_indices,:]
    ))

    valid_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": train_tokens["input_ids"].numpy()[valid_indices],
         "attention_mask": train_tokens["attention_mask"].numpy()[valid_indices],
         "token_type_ids": train_tokens["token_type_ids"].numpy()[valid_indices]},
        data_label[valid_indices,:]
    ))
    
    valid_dataset2 = tf.data.Dataset.from_tensor_slices((
        {"input_ids": valid_tokens["input_ids"].numpy(),
         "attention_mask": valid_tokens["attention_mask"].numpy(),
         "token_type_ids": valid_tokens["token_type_ids"].numpy()},
        data_valid_label
    ))

    # バッチ化、シャッフル
    train_dataset_b = train_dataset.batch(8)
    train_dataset_b = train_dataset_b.shuffle(len(train_indices))
    valid_dataset_b = valid_dataset.batch(8)
    valid_dataset_b = valid_dataset_b.shuffle(len(valid_indices))
    valid_dataset2_b = valid_dataset2.batch(8)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        #"/kaggle/working/model"+str(fold)+"-{epoch:02d}-{val_auc:.2f}.keras"
        "/kaggle/working/model"+str(fold)+".keras"
        ,monitor = "val_loss",direction = "min"
        ,save_best_only = True
        #,period=2
        ,save_weights_only = True
)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.fit(train_dataset_b,
              validation_data=valid_dataset_b,
              epochs=100,
              callbacks=[checkpoint, callback],
              #verbose=0
             )
    
    valid_loss, valid_acc = model.evaluate(valid_dataset2, verbose=0)
    valid_scores.append(valid_acc)

    models.append(model)

cv_score = np.mean(valid_scores)
print(f'CV score: {cv_score}')
```
val_accが上がっていっていることが確認できた。（GPUの利用時間に制限があるので途中で止めた。最終的な精度は不明。）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/542929/6e4a96be-cb6b-8515-0b37-92904e2901ec.png)

# おわりに
KerasでBERTモデルを使った事前学習やファインチューニングを実施した。これでKerasでtransformersライブラリのクラスを使えるようになった気がする。ただ最近、tf.kerasではなく、スタンドアロンのKeras 3.0がリリースされたのでそっちだとまた微妙に書き方とかが違う可能性もあるなと思っている。その辺はまた追々お勉強しますかね。

以上！
