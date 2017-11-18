# Cyclegan-Tensorflow
Simple implementation of Cyclegan by Tensorflow.

CycleGan is a very interesting model which can transfer the data from one domain to the other. Typical use is to transfer the painting style of the photos (like neural style transfer) or transfer the object within the images from one to the other (apple to orange, horse to zebra).

Training Cyclegan does not require explicit paired data. It is a semi-supervised model.

CycleGan paper:https://arxiv.org/abs/1703.10593. 

## Data

Data for training and can be downloaded here: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

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
 
 
Zebra to House:

![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B2.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A2.png)

![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B5.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A5.png)

![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B9.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A9.png)

![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/B13.png) ![img](https://github.com/WoshidaCaiB/Cyclegan-Tensorflow/blob/master/img/A13.png)

The model does not generate results well. During training, I found the Discriminator A (discriminating the images of horse) is very powerful and always defer the fake horse images. This makes the discriminator unable to help generator(horse to zebra) converge. 
