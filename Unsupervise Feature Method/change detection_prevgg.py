

import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage import filters 
from skimage import exposure
from keras import backend as K
from tensorflow.keras.models import Model
import cv2
#Function to retrieve features from intermediate layers
def get_activations(model, layer_idx, X_batch):
    
    partial_model = Model(model.inputs, model.layers[layer_idx].output)
    activations =  partial_model.predict(X_batch)
    return activations

#Function to extract features from intermediate layers
def extra_feat(img_path):
        #Using a VGG19 as feature extractor 
        # 
        # 
        base_model = VGG19(weights='imagenet', include_top=False)


        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        block1_pool_features = get_activations(base_model, 3, x)
        block2_pool_features = get_activations(base_model, 6, x)
        block3_pool_features = get_activations(base_model, 10, x)
        block4_pool_features = get_activations(base_model, 14, x)
        block5_pool_features = get_activations(base_model, 18, x)

        x1 = tf.image.resize(block1_pool_features[0],size=[112, 112])
        x2 = tf.image.resize(block2_pool_features[0],size=[112, 112])
        x3 = tf.image.resize(block3_pool_features[0],size= [112, 112])
        x4 = tf.image.resize(block4_pool_features[0],size= [112, 112])
        x5 = tf.image.resize(block5_pool_features[0],size= [112, 112])

        # Change to only x1, x1+x2,x1+x2+x3..so on, inorder to visualize features from diffetrrnt blocks
        print(x1.shape)
        F = tf.concat(values=[x3, x2, x1, x4, x5],axis=2)
        return F
def main():
  if (len(sys.argv))>3:
    print ("Invalid number of input arguments ")
    exit(0)

  #Two aerial patches with change or No change
  img_path1=sys.argv[1]
  img_path2=sys.argv[2]

  with tf.compat.v1.Session() as sess:

        F1=extra_feat(img_path1) #Features from image patch 1
        F1=tf.square(F1)
        F2=extra_feat(img_path2) #Features from image patch 2
        F2=tf.square(F2)
        d=tf.subtract(F1,F2)
        d=tf.square(d) 
        d=tf.reduce_sum(d,axis=2) 

        dis=d.eval()   #The change map formed showing change at each pixels
        dis=np.resize(dis,[112,112])

        # Calculating threshold using Otsu's Segmentation method
        val = filters.threshold_otsu(dis[:,:])
        hist, bins_center = exposure.histogram(dis[:,:],nbins=256)
        
        plt.title('Change')
        plt.imshow(dis[:,:] < val, cmap='gray', interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def singleImage():
  if (len(sys.argv))>2:
    print ("Invalid number of input arguments ")
    exit(0)

  #Two aerial patches with change or No change
  img_path1=sys.argv[1]

  with tf.compat.v1.Session() as sess:

        F1=extra_feat(img_path1) #Features from image patch 1
        F1=tf.square(F1)
        d=tf.square(F1) 
        d=tf.reduce_sum(d,axis=2) 

        dis=d.eval()   
        dis=np.resize(dis,[112,112])

        # Calculating threshold using Otsu's Segmentation method
        val = filters.threshold_otsu(dis[:,:])
        hist, bins_center = exposure.histogram(dis[:,:],nbins=256)
        
        plt.title('Change')
        plt.imshow(dis, cmap='gray', interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()
    #singleImage()
