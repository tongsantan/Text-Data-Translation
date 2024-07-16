## 5. Update the components

import os
import sys
from src.exception import CustomException
from src import logger
import pandas as pd
from dataclasses import dataclass
from src.utils.common import read_text, to_lines 
import numpy as np
from numpy import array
import string
from src.entity.config_entity import DataIngestionConfig

## 5. Update the components

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        '''
        This function is responsible for data ingestion
        
        '''
        logger.info("Data Ingestion") 
        
        try:
            logger.info("Reading the data")

            data = read_text(self.config.input_data_path)

            fra_eng = to_lines(data)

            fra_eng = array(fra_eng)

            fra_eng = fra_eng[:50000,:]

            # Remove punctuation
            fra_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,0]]
            fra_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,1]]

            # convert text to lowercase
            for i in range(len(fra_eng)):
                fra_eng[i,0] = fra_eng[i,0].lower()
                fra_eng[i,1] = fra_eng[i,1].lower()

            os.makedirs(os.path.dirname(self.config.word_array_path),exist_ok=True)

            logger.info("Saving the data")

            np.save(self.config.word_array_path, fra_eng)

            logger.info("Ingestion of the data is completed")

            return fra_eng
        
        except Exception as e:
            raise CustomException(e,sys)  