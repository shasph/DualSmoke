DualSmoke: 
Sketch-Based Smoke Illustration Design withTwo-Stage Generative Model
(under submission)

#
This is the source code of paper, DualSmoke:Sketch-Based Smoke Illustration Design withTwo-Stage Generative Model.
![image](https://user-images.githubusercontent.com/4180028/185010140-59cfa7c7-f463-4e7e-ae46-fda8c6f51b04.png)

## preparation
1. install mantaflow
```
sudo apt-get install cmake g++ git python3-dev qt5-qmake qt5-default
```

2. commands for mantaflow
```
(cd Sketch2Smoke/mantaflow)
mkdir build
cd build
cmake .. -DGUI=ON -DNUMPY=ON
make -j4
```
OPENMP can be used for high-performance, TBB to be valided.
</br>

3. move /build to Sketch2Smoke/scripts/client
```
(cd Sketch2Smoke)
mv mantaflow/build scripts/client/
```

## run pre-trained model
1. install cuda for pytorch

2. install flask
```
python3 -m pip install flask
```

## run tips
1. open terminal in Sketch2Smoke/scripts/server/pi2pix with following command
```
python3 app.py
```

2. open terminal in Sketch2Smoke/scripts/client with following command
```
python3 exe.py
```

## pre-trained model
* Sketch2LCS : Sketch2Smoke/scripts/server/pix2pix/checkpoints/skeleton2lcs/latest_net_G.pth
* LCS2Vel : Sketch2Smoke/scripts/server/pix2pix/checkpoints/lcs2vel_pix2pix/latest_net_G.pth

