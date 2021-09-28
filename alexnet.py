
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


from tensorflow.math import confusion_matrix
from tensorflow.keras import backend as K  

import matplotlib.pyplot as plt
import numpy as np


class AlexNet:
    
    '''
    
    ALEXNET MODEL ARCHITECTURE WITH UTILITIES
    
    '''
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def build_model(self):
        
        '''
        BUILDS THE ARCHITECTURE. DISPLAYS THE MODEL AND PARAMETERS
        '''
        
        model=Sequential()
        #1 conv layer
        model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="valid",activation="relu",input_shape=(227,227,3)))

        #1 max pool layer
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(BatchNormalization())
        #2 conv layer
        model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid",activation="relu"))
        #2 max pool layer
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(BatchNormalization())
        #3 conv layer
        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))
        #4 conv layer
        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))
        #5 conv layer
        model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))
        #3 max pool layer
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.4))
        #1 dense layer
        model.add(Dense(4096,input_shape=(227,227,3),activation="relu"))
        model.add(Dropout(0.4))
        #2 dense layer
        model.add(Dense(4096,activation="relu"))
        

        #output layer
        model.add(Dense(2,activation="sigmoid"))
       
        model.summary()
       
        return model


    def compile_model(self, model, optimizer, loss, metrics):
        '''
        COMPILES THE MODEL.
        '''
        
        model.compile(optimizer= optimizer, loss=loss, metrics=metrics)
        
    def fit_model(self, model, batch_size, epochs):
        
        '''
        FITS MODEL TO THE TRAINING SET AND VALIDATES THE DATA.
        '''
        
        return model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(self.X_val, self.y_val))
    
    def evaluate_model(self, model):
        
        '''
        EVALUATIONS ON TEST DATA
        '''
        
        loss, accuracy, f1_score, precision, recall = model.evaluate(self.X_test, self.y_test, verbose=0)
        
        return loss, accuracy, f1_score, precision, recall
    
    def plot_confusion_matrix(self, model):
        
        
        '''
        PRINTS CONFUSION MATRIX FORM tf.math.confusion_matrix
        '''
        
        
        y_pred = model.predict(self.X_test)
        y_predict = []
        for i in range(0,self.X_test.shape[0]):
            if y_pred[i][0] >= y_pred[i][1]:
                y_predict.append(0)
            elif y_pred[i][0] < y_pred[i][1]:
                y_predict.append(1)
        confusionMatrix = confusion_matrix(labels=self.y_test, predictions=y_predict, num_classes=2 )
        print(confusionMatrix)
        
                 
    
    def plot_history(self, history):
        
        
        '''
        PLOTS ALL EVALUATION METRICS OF TRAINING AND VALIDATION SETS. 
        '''

        # History for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


        # History for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
         
        # History for f1 score
        plt.plot(history.history['f1_m'])
        plt.plot(history.history['val_f1_m'])
        plt.title('model F1 score')
        plt.ylabel('F1 score')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
         
        # History for precision
        plt.plot(history.history['precision_m'])
        plt.plot(history.history['val_precision_m'])
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
         
         
        # History for recall
        plt.plot(history.history['recall_m'])
        plt.plot(history.history['val_recall_m'])
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    

        
    def predictions(self, model, categories, verbose = False):
        
        '''
        PREDICTS ON RANDOM TEST IMAGES.
        '''
        
        if verbose:
            print(categories)
            print(model.predict(self.X_test))

        pred = model.predict(self.X_test)
        n = 0
        r = np.random.randint(0 , self.X_test.shape[0] , 9)
        for rand_img in r:
            n+=1
            img_predicted = pred[rand_img]
            prediction = img_predicted.max()
            for i in range(0,len(categories)):
                if prediction == img_predicted[i]:
                    class_name = categories[i]
    
            
            plt.imshow(self.X_test[rand_img])
            plt.axis('off')
            plt.title(class_name)
            plt.show()
        
    
    def recall_m(self, y_true, y_pred):
        '''
        CALCULATES RECALL
        '''
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        if recall > float(1):
            return float(1)
        return recall

    def precision_m(self, y_true, y_pred):
        '''
        CALCULATES PRECISION
        '''
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        '''
        CALCULATES F1_SCORE
        '''
        
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*(precision*recall)/(precision+recall+K.epsilon())       
     