import alexnet
import dataprocessor
import time
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Removes tf-gpu logs


if __name__ == "__main__":
    
    '''
    Define common hyperparameter values 
    '''
    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    '''
    Adam optimizer finds local minima thus our model error does not minimizes.
    SGD works very well with the following hyperparameter tuning
    '''
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    loss="sparse_categorical_crossentropy" # Better than Binary_crossentropy
    
    
    epochs = 50 # Best epoch numbers
    batch_size = 64 # least fluctuations in validation through this batch size
    
    
    
    print("\n************************************")
    print("    ALEXNET WITH GRAPHS AS INPUT     ")
    print("************************************")
    
    
    # path to the augmented plots
    path_G = r"C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/graph_data/Aug_data" 
    random_seed = 42
    
    
    # Defining categories to predict
    categories = os.listdir(path_G)
    categories = categories[:2]
    print("List of categories = ",categories,"\n\nNo. of categories = ", len(categories))
    
    

    '''
    DATA PRE-PROCESSING
    '''
    
    data_G = dataprocessor.DataProcessor(path_G, categories)
    
    # Loading plots and getting labels. Converting plots to numpy arrays.
    images_G, labels_G = data_G.load_images_and_labels() 
    # Visualizing plots
    data_G.display_random_images(images_G, labels_G) 
    # Shuffling for better performance of model
    images_G, labels_G = data_G.shuffle_images(images_G, labels_G) 
    # Normalizing to the convention of 255
    images_G, labels_G = data_G.normalize_images(images_G, labels_G)
    # Spliting data with the mentioned ratio
    X_train_G, y_train_G, X_val_G, y_val_G, X_test_G, y_test_G = data_G.data_split(images_G, 
                                                                     labels_G, 
                                                                     train_ratio, 
                                                                     validation_ratio, 
                                                                     test_ratio
                                                                     )
    
    '''
    MODEL TRAINING AND EVALUATION
    '''
    start = time.time()
    # Loading our AlexNet model class
    alexnet_G = alexnet.AlexNet(X_train_G, y_train_G, X_val_G, y_val_G, X_test_G, y_test_G)
    # Building AlexNet
    model_G = alexnet_G.build_model()
    # We will use accuracy, recall, precision and F1 score as our evaluation metrics
    metrics_G = ['acc', alexnet_G.f1_m, alexnet_G.precision_m, alexnet_G.recall_m]
    # Compiling the model
    alexnet_G.compile_model(model_G, sgd, loss, metrics_G)
    # Fitting model to our data
    history_G = alexnet_G.fit_model(model_G, batch_size, epochs)
    end = time.time()
    
    print("\n Time taken: ", end - start)
    
    '''
    PLOTTING TRAINING AND VALIDATION DATA RESULTS
    '''
    
    
    # Ploting training and validation evaluation metrics
    alexnet_G.plot_history(history_G)
    # Prediction on random plots
    alexnet_G.predictions(model_G, categories)
    
    
    
    
    '''
    
    EACH STEP REMAINS SAME FOR RECURRENCE PLOT CLASSIFICATION
    Each hyperparameter value is same so that there can be a 
    fair comparison between two candidate plots.
    
    '''
    
    
    print("\n***********************************************")
    print("    ALEXNET WITH RECURRENCE PLOTS AS INPUT       ")
    print("*************************************************")
    
    
    path_R = r"C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/exoPlanet_data_1.1/Aug_data"
    random_seed = 42
    
    
    
    categories = os.listdir(path_R)
    categories = categories[:2]
    print("List of categories = ",categories,"\n\nNo. of categories = ", len(categories))
    
    
    
    '''
    DATA PRE-PROCESSING
    '''
    
    
    data_R = dataprocessor.DataProcessor(path_R, categories)
    
    images_R, labels_R = data_R.load_images_and_labels()
    
    data_R.display_random_images(images_R, labels_R)
    
    images_R, labels_R = data_R.shuffle_images(images_R, labels_R)
    
    images_R, labels_R = data_R.normalize_images(images_R, labels_R)
    
    X_train_R, y_train_R, X_val_R, y_val_R, X_test_R, y_test_R = data_R.data_split(images_R, 
                                                                     labels_R, 
                                                                     train_ratio, 
                                                                     validation_ratio, 
                                                                     test_ratio
                                                                     )
    
    '''
    MODEL TRAINING AND EVALUATION
    '''
    
    start = time.time()
    
    alexnet_R = alexnet.AlexNet(X_train_R, y_train_R, X_val_R, y_val_R, X_test_R, y_test_R)
    
    model_R = alexnet_R.build_model()

    metrics_R = ['acc', alexnet_R.f1_m, alexnet_R.precision_m, alexnet_R.recall_m]
    
    alexnet_R.compile_model(model_R, sgd, loss, metrics_R)
    
    history_R = alexnet_R.fit_model(model_R, batch_size, epochs)
    
    end = time.time()
    
    print("\n Time taken: ", end - start)
    
    '''
    PLOTTING TRAINING AND VALIDATION DATA RESULTS
    '''
    
    
    alexnet_R.plot_history(history_R)
    
    alexnet_R.predictions(model_R, categories)
    
    
    '''
    EVALUATION ON THE TEST SETS
    '''
    
    
    loss_tr, accuracy_tr, f1_score_tr, precision_tr, recall_tr = alexnet_R.evaluate_model(model_R)
    
    
    print("\n*****************************************************")
    print("PERFORMANCE ON TEST SET  WITH RECURRENCE PLOTS AS INPUT")
    print("*******************************************************")
    
    print("\nAccuracy: ",accuracy_tr)
    print("\nLoss: ",loss_tr)
    print("\nF1 ScorE: ",f1_score_tr)
    print("\nPrecision: ",precision_tr)
    print("\nRecall: ",recall_tr)
    
    # Getting Confusion Marix
    print("\nConfusion matrix:")
    alexnet_G.plot_confusion_matrix(model_G)
    
    loss_tg, accuracy_tg, f1_score_tg, precision_tg, recall_tg = alexnet_G.evaluate_model(model_G)
    
    print("\n*****************************************************")
    print("PERFORMANCE ON TEST SET WITH LIGHT CURVE PLOTS AS INPUT")
    print("*******************************************************")
    
    print("\nAccuracy: ",accuracy_tg)
    print("\nLoss: ",loss_tg)
    print("\nF1 ScorE: ",f1_score_tg)
    print("\nPrecision: ",precision_tg)
    print("\nRecall: ",recall_tg)
    
    # Getting Confusion Marix
    print("\nConfusion matrix:")
    alexnet_G.plot_confusion_matrix(model_G)
