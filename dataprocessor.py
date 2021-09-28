import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataProcessor:
    '''
    DATA PRE-PROCESSING METHODS
    '''
    def __init__(self, path, categories):
        self.path = path
        self.categories = categories
    
    def load_images_and_labels(self):
        '''
        
        LOADS INPUT PLOTS AND EXTRACTS LABELS. CONVERTS LABELS AND PLOTS INTO
        NUMPY ARRAYS.

        '''
        images=[]
        labels=[]
        for index, category in enumerate(self.categories):
            for image_name in os.listdir(self.path+"/"+category):
                img = cv2.imread(self.path+"/"+category+"/"+image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_array = Image.fromarray(img, 'RGB')
            
                #resize image to 227 x 227 because the input image resolution for AlexNet is 227 x 227
                resized_img = img_array.resize((227, 227))
                
                images.append(np.array(resized_img))
            
                labels.append(index)
        print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
        #converting images to np.array
        images = np.array(images)
        labels = np.array(labels)
        print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)
        print(type(images),type(labels))
        return images, labels
        
    
    def display_random_images(self, images, labels):
        
        '''
        DISPLAYING RANDOM INPUT PLOTS
        '''
        
        plt.figure(1 , figsize = (19 , 10))
        n = 0 
        for i in range(9):
            n += 1 
            r = np.random.randint(0 , images.shape[0] , 3)
        
            plt.imshow(images[r[2]])
        
            plt.title('Planets : {}'.format(labels[r[2]]))
            plt.xticks([])
            plt.yticks([])
        
            plt.show()
        
    
    def shuffle_images(self, images, labels, random_seed = 42):
        
        '''
        
        SHUFFLE IMAGES FOR BETTER MODEL TRAINING
        
        '''
        
        
        #1-step in data shuffling
        #get equally spaced numbers in a given range
        n = np.arange(images.shape[0])
        print("'n' values before shuffling = ",n)

        #shuffle all the equally spaced values in list 'n'
        np.random.seed(random_seed)
        np.random.shuffle(n)
        print("\n'n' values after shuffling = ",n)

        #2-step in data shuffling
        #shuffle images and corresponding labels data in both the lists
        images = images[n]
        labels = labels[n]
        
        print("Images shape after shuffling = ",images.shape,"\nLabels shape after shuffling = ",labels.shape)
        return images, labels
    
    
    def normalize_images(self, images, labels):
        
        '''
        NORMALIZES THE DATA TO CONVENTIONAL 255
        '''
        

        images = images.astype(np.float32)
        labels = labels.astype(np.int32)
        images = images/255
        print("Images shape after normalization = ",images.shape)
        
        return images, labels
    
    def data_split(self, images, labels, train_ratio, validation_ratio, test_ratio):
        
        '''
        SPLITS INTO TRAIN, VALIDATION AND TEST SET
        '''
        
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=1 - train_ratio)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 


        print("Shapes after the split:")
        print("\nx_train shape = ",X_train.shape)
        print("y_train shape = ",y_train.shape)
        print("\nx_val shape = ",X_val.shape)
        print("y_val shape = ",y_val.shape)
        print("\nx_test shape = ",X_test.shape)
        print("y_test shape = ",y_test.shape)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
   