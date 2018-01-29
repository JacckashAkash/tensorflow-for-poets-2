# Overview
This Repo is somewhat changed by Ritesh for the Series - [IMAGE CLASSIFICATION WITH TENSORFLOW](https://www.youtube.com/playlist?list=PLtl9EQhH8dm3BaqXJBrUvVaITzM0xxD-t) on YouTube.

This repo contains code for the "TensorFlow for poets 2" series of codelabs for Classifying Images in five different categories.

There are multiple versions of this codelab depending on which version 
of the tensorflow libraries you plan on using:

* For [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) the new, ground up rewrite targeted at mobile devices
  use [this version of the codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite) 
* For the more mature [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro) use 
  [this version of the codealab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).

* SETUP:

 Install Tensor Flow:
    In the Terminal/CMD, hit the following:
    
       pip install tensorflow
      
 Clone the git repository
    All the code used in this codelab is contained in this git repository. Clone the repository and cd into it. This is where we will be
    working.
      
       git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
       cd tensorflow-for-poets-2
      
* DOWNLOAD the training images:

  Before you start any training, you'll need a set of images to teach the model about the new classes you want to recognize. 
  
      http://download.tensorflow.org/example_images/flower_photos.tgz
    
   To Download these images, Use [MY CODE-File Downloader.py](https://github.com/MauryaRitesh/Python/blob/master/file_downloader-progress_bar.py) OR simply paste the above link in the Address Bar of your Favorite Web Browser!
    

* (RE)TRAINING the Network:

  The retrain script can retrain either Inception V3 model or a MobileNet. In this series, we will use an Inception V3 model. The principal
  difference is that Inception V3 is optimized for accuracy, while the MobileNets are optimized to be small and efficient, at the cost 
  of some accuracy.
    The Script Downloads the pre-trained Inception_v3 model and adds a FINAL Layer to it by TRAINING it on OUR Images.
    
   To Start (Re)Training the Network, hit the following in the Terminal/CMD:
   
      python -m scripts.retrain
      --bottleneck_dir=tf_files/bottlenecks
      --model_dir=tf_files/models
      --summaries_dir=tf_files/training_summaries
      --output_graph=tf_files/retrained_graph.pb
      --how_many_training_steps=4000
      --output_labels=tf_files/retrained_labels.txt
      --architecture=inception_v3
      --image_dir=tf_files/flower_photos
    
* Using the RETRAINED MODEL:
  
  Classifying an image

   The codelab repo also contains a copy of tensorflow's [label_image.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py) 
   example, which you can use to test your network.
    
   To Classify any Image, hit the Following:
   
      python -m scripts.label_image \
      --graph=tf_files/retrained_graph.pb  \
      --image=[Your Image HERE]
    
   This will Classify the Given Image on the basis of the Training. Note that, it'll try to classify any type of image, say a person     
   into these five types of flowers. So, for now it's the end and please tell me which type of your face looks after being classified by 
   the Retrained Classifier!!!
