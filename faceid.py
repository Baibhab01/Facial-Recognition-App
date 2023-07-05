#Import Kivy Dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger

#import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import numpy as np
import os

#Build APP and layout
class CamApp(App):

    def build(self):
        #Main LAyout Component
        self.web_cam = Image(size_hint=(1, .9))
        self.button = Button( text='Capture', on_press = self.verify, size_hint=(1, .1))
        self.verfication_label = Label(text='Verification Uninitiated', size_hint=(1, .1))

        # add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verfication_label)

        #Load Tensorflow/keras Model 
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects= {'L1Dist':L1Dist})

        #Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        
        return layout
    
    #Run continuosly to get webcam feed
    def update(self, *args):

        #read the frame from openvc
        ret ,frame = self.capture.read()
        frame = frame[120:120+250, 150:200+250, :]

        #flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture =Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    #Load image from file and convert 100x100 pixels
    def preprocess(self,file_path):
        #Read the image from the file path
        byte_img =tf.io.read_file(file_path)
        #Load the image
        img = tf.io.decode_jpeg(byte_img)
    
        #Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
    
        #Scaling the image between 0 and 1 to help in gradient decent and optomize our model
        img = img/ 255.0
        return img
    
    #Verification function to verify
    def verify(self , *args):
        #Specify Threshold
        detection_threshold =   0.6
        verification_threshold= 0.68

        #Capture input img from webcam
        SAVE_PATH = os.path.join('application_data','input_images', 'input_images.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 150:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
 
        #Build results array
        results= []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_images', 'input_images.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            #Make Predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
            
        #Detection Threshold : Metric above which a prediction is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        #Verification Threshold : Proportion of positive predictions /total positive samples
        verification = detection/ len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold
        
        #Set Verification text
        self.verfication_label.text = 'Verified' if verified == True else 'Unverified'

        #Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified
        

if __name__ == '__main__':
    CamApp().run()