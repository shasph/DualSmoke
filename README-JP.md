# Sketch2Smoke

## mantaflow動作準備
1. mantaflowのインストールのためのパッケージをインストール
```
sudo apt-get install cmake g++ git python3-dev qt5-qmake qt5-default
```

2. mantaflowディレクトリ以下でコマンドを実行
```
(cd Sketch2Smoke/mantaflow)
mkdir build
cd build
cmake .. -DGUI=ON -DNUMPY=ON
make -j4
```

ただし、シミュレーション高速化のためにOPENMPか、TBBの有効化も可能。
</br>

3. buildディレクトリをSketch2Smoke/scripts/clientに移動
```
(cd Sketch2Smoke)
mv mantaflow/build scripts/client/
```

## 学習済みモデル動作準備
1. pytorchを使用するために、cuda及び対応するpytorchをインストール

2. flaskをインストール
```
python3 -m pip install flask
```

## 実行手順
1. Sketch2Smoke/scripts/server/pi2pixディレクトリ内でターミナルを開き、以下のコマンドを実行(ターミナルは閉じない)
```
python3 app.py
```

2. Sketch2Smoke/scripts/clientディレクトリ内でターミナルを開き、以下のコマンドを実行
```
python3 exe.py
```


## システム概要
基本的なシステムの動きとしては、  
ユーザスケッチ→学習済みモデルによる速度場生成→ガイド付きシミュレーション  
となる。

以下の項では、  

* ユーザスケッチ
* 学習済みモデルによる速度場生成
* ガイド付きシミュレーション  

及び、これらを動かすための環境と研究手順について説明する。  

</br>
</br>

## 実行環境
</br>
</br>

## ユーザスケッチ

</br>
</br>

## 学習済みモデルについて
学習済みモデルは以下のパスに保存されている。  
* Sketch2LCS : Sketch2Smoke/scripts/server/pix2pix/checkpoints/skeleton2lcs/latest_net_G.pth
* LCS2Vel : Sketch2Smoke/scripts/server/pix2pix/checkpoints/lcs2vel_pix2pix/latest_net_G.pth


学習済みモデルを使用して速度場を生成する。基本的には、ユーザによるスケッチを受取り、  
* Sketch2LCSにスケッチを入力→LCSを出力→LCS2VelにLCSを入力→Velを出力

という流れとなる。私の実行環境は、都合上、別サーバで動かしていたためhttp通信によってファイルの送受信を行っていた。サーバについては、  
Sketch2Smoke/scripts/server/pix2pix/app.py  
を参照のこと。
