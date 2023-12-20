from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt



def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def train_and_evaluate_oos(X_train, y_train, batch_size, epochs, out_of_sample = None, lr = 3e-5, name = None):
    '''
    Inputs:
    X_train: nparray, (images, dim1, dim2, channels)
    y_train: nparray, (masks, dim1, dim2, 1)
    out_of_sample: list, a list of out of sample indices

    
    Returns:
    -model
    -plots of out of sample predictions and their masks
    
    '''
    
    oos = out_of_sample

    if oos is not None:
        X_val = X_train[oos,:,:,:]
        y_val = y_train[oos,:,:,:]
        X_train = np.delete(X_train,oos,0)
        y_train = np.delete(y_train,oos,0)
   
    
    #Build the model
    with tf.device('/gpu:0'):
        inputs = tf.keras.layers.Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        
        kernel_size = (3,3)
        #Contraction path
        b11 = tf.keras.layers.BatchNormalization()(inputs)
        c1 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b11)
        c1 = tf.keras.layers.Dropout(0.2)(c1)
        b12 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b12)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        b21 = tf.keras.layers.BatchNormalization()(p1)
        c2 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b21)
        c2 = tf.keras.layers.Dropout(0.2)(c2)
        b22 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b22)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        
        b31 = tf.keras.layers.BatchNormalization()(p2)
        c3 = tf.keras.layers.Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b31)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        b32 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b32)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        b41 = tf.keras.layers.BatchNormalization()(p3)
        c4 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b41)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        b42 = tf.keras.layers.BatchNormalization()(c4)
        c4 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b42)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
        
        b51 = tf.keras.layers.BatchNormalization()(p4)
        c5 = tf.keras.layers.Conv2D(256, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b51)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        b52 = tf.keras.layers.BatchNormalization()(c5)
        c5 = tf.keras.layers.Conv2D(256, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b52)

        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(128, kernel_size, strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        b61 = tf.keras.layers.BatchNormalization()(u6)
        c6 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b61)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        b62 = tf.keras.layers.BatchNormalization()(c6)
        c6 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b62)
        
        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        b71 = tf.keras.layers.BatchNormalization()(u7)
        c7 = tf.keras.layers.Conv2D(64,kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b71)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        b72 = tf.keras.layers.BatchNormalization()(c7)
        c7 = tf.keras.layers.Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b72)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        b81 = tf.keras.layers.BatchNormalization()(u8)
        c8 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b81)
        c8 = tf.keras.layers.Dropout(0.2)(c8)
        b82 = tf.keras.layers.BatchNormalization()(c8)
        c8 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b82)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        b91 = tf.keras.layers.BatchNormalization()(u9)
        c9 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b91)
        c9 = tf.keras.layers.Dropout(0.2)(c9)
        b92 = tf.keras.layers.BatchNormalization()(c9)
        c9 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b92)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=Adam(lr=lr), loss = dice_coef_loss, metrics=['accuracy'])
        
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 30, restore_best_weights = True)
        if oos is not None:
            history = model.fit(x = X_train,
                                y = y_train,
                                epochs = epochs,
                                batch_size = batch_size,
                                verbose = 1, 
                                validation_data = (X_val,y_val),
                                callbacks = [early_stopping_cb])
            X_new = X_val
            if len(oos) > 1:
                fig, axs = plt.subplots(len(oos),3, figsize=(25,40))
                for i in range(len(oos)):
                    axs[i,0].imshow(X_new[i,:,:,0])
                    axs[i,1].imshow(model.predict(X_new[[i]])[0,:,:,0])
                    axs[i,2].imshow(y_val[i,:,:,0]) 
            if len(oos) == 1:
                fig, axs = plt.subplots(3, figsize=(10,15))
                for i in range(len(oos)):
                    axs[0].imshow(X_new[i,:,:,0])
                    axs[1].imshow(model.predict(X_new[[i]])[0,:,:,0])
                    axs[2].imshow(y_val[i,:,:,0]) 

        else:
            history = model.fit(x = X_train,
                                y = y_train,
                                epochs = epochs,
                                batch_size = batch_size,
                                verbose = 1)
        if name is not None:
            model.save("{}.h5".format(name))
        return model,history


def train_and_evaluate_oos_augmented(X_train, y_train, batch_size, epochs, steps_per_epoch, out_of_sample = None, lr = 3e-5, name = None):
    '''
    Inputs:
    out_of_sample: list, a list of out of sample indicides
    
    Returns:
    plots of out of sample predictions and their masks
    
    '''
    
    oos = out_of_sample
    data_gen_args = dict(rotation_range = 180,
                        horizontal_flip = True,
                        vertical_flip = True,
                        shear_range = 0.2,
                        zoom_range = 0.6,
                        width_shift_range = (0.6,1.5))
    print(X_train.shape)
    if oos is not None:
        X_train = np.delete(X_train,oos,0)
        y_train = np.delete(y_train,oos,0)
        X_val = X_train[oos,:,:,:]
        y_val = y_train[oos,:,:,:]
    print(X_train.shape)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 2 #Provide the same seed and keyword arguments to the fit and flow methods
    image_datagen.fit(X_train, augment = True, seed = seed)
    image_datagen.fit(y_train, augment = True, seed = seed)
    image_generator= image_datagen.flow(X_train, seed = seed)
    mask_generator = mask_datagen.flow(y_train, seed = seed)
    train_generator = zip(image_generator,mask_generator)
    
    #Build the model
    with tf.device('/gpu:0'):
        inputs = tf.keras.layers.Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        
        kernel_size = (3,3)
        #Contraction path
        b11 = tf.keras.layers.BatchNormalization()(inputs)
        c1 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b11)
        c1 = tf.keras.layers.Dropout(0.2)(c1)
        b12 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b12)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        b21 = tf.keras.layers.BatchNormalization()(p1)
        c2 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b21)
        c2 = tf.keras.layers.Dropout(0.2)(c2)
        b22 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b22)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        
        b31 = tf.keras.layers.BatchNormalization()(p2)
        c3 = tf.keras.layers.Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b31)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        b32 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b32)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        b41 = tf.keras.layers.BatchNormalization()(p3)
        c4 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b41)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        b42 = tf.keras.layers.BatchNormalization()(c4)
        c4 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b42)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
        
        b51 = tf.keras.layers.BatchNormalization()(p4)
        c5 = tf.keras.layers.Conv2D(256, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b51)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        b52 = tf.keras.layers.BatchNormalization()(c5)
        c5 = tf.keras.layers.Conv2D(256, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b52)

        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(128, kernel_size, strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        b61 = tf.keras.layers.BatchNormalization()(u6)
        c6 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b61)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        b62 = tf.keras.layers.BatchNormalization()(c6)
        c6 = tf.keras.layers.Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b62)
        
        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        b71 = tf.keras.layers.BatchNormalization()(u7)
        c7 = tf.keras.layers.Conv2D(64,kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b71)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        b72 = tf.keras.layers.BatchNormalization()(c7)
        c7 = tf.keras.layers.Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b72)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        b81 = tf.keras.layers.BatchNormalization()(u8)
        c8 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b81)
        c8 = tf.keras.layers.Dropout(0.2)(c8)
        b82 = tf.keras.layers.BatchNormalization()(c8)
        c8 = tf.keras.layers.Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b82)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        b91 = tf.keras.layers.BatchNormalization()(u9)
        c9 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b91)
        c9 = tf.keras.layers.Dropout(0.2)(c9)
        b92 = tf.keras.layers.BatchNormalization()(c9)
        c9 = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(b92)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=Adam(lr=lr), loss = dice_coef_loss, metrics=['accuracy'])
        
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 30, restore_best_weights = True)
        if oos is not None:
            history = model.fit(train_generator,
                      steps_per_epoch=steps_per_epoch, 
                      epochs = epochs,
                      batch_size = batch_size,
                      verbose = 2, 
                      validation_data = (X_val,y_val),
                      callbacks = [early_stopping_cb])
            X_new = X_train[oos,:,:,:]
            fig, axs = plt.subplots(len(oos),3, figsize=(25,40))
            for i in range(len(oos)):
                axs[i,0].imshow(X_new[i,:,:,0])
                axs[i,1].imshow(model.predict(X_new[[i]])[0,:,:,0])
                axs[i,2].imshow(y_train[oos[i]][:,:,0]) 
        else:
            history = model.fit(train_generator,
                      steps_per_epoch=steps_per_epoch, 
                      epochs = epochs,
                      batch_size = batch_size,
                      verbose = 2)
        if name is not None:
            model.save("{}.h5".format(name))
        return model,history