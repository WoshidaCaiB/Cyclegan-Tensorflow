# Cyclegan-Tensorflow
Simple implementation of Cyclegan by Tensorflow

paper:https://arxiv.org/abs/1703.10593

Data can be downloaded here: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

Download data and unzip to your local path. The data contains four components: trainA, trainB, testA and testB

## Run model

To train model:

   python train.py --train_file=local path where data is stoted --model_file=path where you want to store the model (if you do not specify    this, model will be saved in file: CycleGan.ckpt under current working path)


To generate results:

   python generate.py --test_file=local path where data is stored --model_file=the path where model is saved
   
## Results:
 
 Due to the constrain of resources, I only implemented experiment on horse to zebra.
 
 Horse to Zebra
 
 ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A1.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B1.png)
 
 ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A4.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B4.png)
 
 ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A6.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B6.png)
 
 ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A7.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B7.png)
 
