import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from transformers import AutoTokenizer
from src.utils.common import save_tokenizer, tokenization, encode_sequences
from src.exception import CustomException
from src import logger
import os
import sys
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataTransformationConfig

## 5. Update the components

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config=config
        
    def tokenizing(self):
        '''
        This function is responsible for tokenizing the data
        
        '''
        logger.info("Loading the array")

        fra_eng = np.load(self.config.word_array_path)

        logger.info("Tokenzing the text data") 

        # prepare english tokenizer
        eng_tokenizer = tokenization(fra_eng[:, 0])

        eng_length = 8
        
        # prepare french tokenizer
        fra_tokenizer = tokenization(fra_eng[:, 1])

        fra_length = 8
        
        save_tokenizer(self.config.eng_tokenizer_data_path, eng_tokenizer)
        save_tokenizer(self.config.fra_tokenizer_data_path, fra_tokenizer)
        
        # split data into train and test set
        train, test = train_test_split(fra_eng, test_size=0.2, random_state = 12)
        
        # prepare training data
        X_train = encode_sequences(fra_tokenizer, fra_length, train[:, 1])
        y_train = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
        
        # prepare validation data
        X_test = encode_sequences(fra_tokenizer, fra_length, test[:, 1])
        y_test = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

        logger.info(f"Saving the train and test datasets.")

        np.save(self.config.test_array_path, test)
        np.save(self.config.X_train_array_path, X_train)
        np.save(self.config.y_train_array_path, y_train)
        np.save(self.config.X_test_array_path, X_test)
        np.save(self.config.y_test_array_path, y_test)

        return(
                self.config.test_array_path,
                self.config.X_train_array_path,
                self.config.y_train_array_path,
                self.config.X_test_array_path,
                self.config.y_test_array_path
            )
