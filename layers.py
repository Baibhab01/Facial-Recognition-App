# Custom L1 Distance Layer Module

#inport dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

#custom L1 distance layer from  Jupeyter Notebook
class L1Dist(Layer):

     ##init method - inheritance
    def __init__(self,**kwargs):
        super().__init__()
        
    ##Passing two streams of images 1-anchor 2-positve/negative for similarity comparison
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) #by subtraction