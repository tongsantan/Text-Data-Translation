import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from src.utils.common import load_tokenizer, define_model, save_object
import os
import sys
from dataclasses import dataclass
import pickle
import warnings
warnings.filterwarnings("ignore")
from src.exception import CustomException
from src import logger
from keras import optimizers
from src.entity.config_entity import ModelTrainerConfig

## 5. Update the components

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config=config

    def initiate_model_trainer(self):
        '''
        This function is responsible for model training
        
        '''
        try:
            
            logger.info(f"Loading the train datasets")
            X_train = np.load(self.config.X_train_array_path)
            y_train = np.load(self.config.y_train_array_path)

            eng_tokenizer = load_tokenizer(self.config.eng_tokenizer_data_path)

            eng_vocab_size = len(eng_tokenizer.word_index) + 1
            
            eng_length = 8
            
            print('English Vocabulary Size: %d' % eng_vocab_size)
            
            fra_tokenizer = load_tokenizer(self.config.fra_tokenizer_data_path)

            fra_vocab_size = len(fra_tokenizer.word_index) + 1
            
            fra_length = 8
            
            print('French Vocabulary Size: %d' % fra_vocab_size)

            # model compilation
            model = define_model(fra_vocab_size, eng_vocab_size, fra_length, eng_length, 512)

            rms = optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # use EarlyStopping in the case when val_accuracy reaches a plateau (not improving much)
            es = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                restore_best_weights=True, 
                patience=10, 
                min_delta = 0.001)
            
            # create model checkpoint callback to save the best model checkpoint
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath="best_checkpoint",
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)

            logger.info(f"Training the model") 
            # train model
            model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1),
                      epochs=30, batch_size=512, validation_split = 0.2,callbacks=[es, model_checkpoint_callback], 
                      verbose=1)
            
            logger.info(f"Saving the trained model")
            save_object(
                    file_path=self.config.trained_model_file_path,
                    obj=model
                )
            
            return model
        
        except Exception as e:
            raise CustomException(e,sys)    
