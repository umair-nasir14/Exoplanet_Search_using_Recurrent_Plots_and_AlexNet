from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


'''
   This file augments the data. It takes one image and applies augmentation
   with rotation, shifting process, applying shear process and horizontal 
   fliping. It repeats the process for 65 times to create 2400 images.
'''

for i in range(5,42):
    datagen = ImageDataGenerator(
        rotation_range=0.4,
        width_shift_range=0.002,
        height_shift_range=0.002,
        shear_range=0.002,
        horizontal_flip=True,
        fill_mode='nearest')
    img = load_img(r'C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/original_data/Train/aug/observation' + str(i) + '.png')  
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)  
    print(x.shape)
    c = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=r'C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/original_data/Train/aug/', save_prefix='recu_obs', save_format='png'):
        c += 1
        if c > 65: #65 for light curve plot
            break  # otherwise the generator would loop indefinitely
            
            

    
    
    