import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
import os
import sys
from src.utils.common import load_tokenizer, load_object, get_word
from src.exception import CustomException
from src import logger
from src.entity.config_entity import ModelEvaluationConfig

## 5. Update the components

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config=config


    def evaluate_model(self):
        '''
        This function is responsible for testing model on unseen datasets
        '''
        try:
            logger.info("Loading model, tokenizer and test dataset")
            model = load_object(self.config.model_data_path)

            test = np.load(self.config.test_array_path)
            
            X_test = np.load(self.config.X_test_array_path)

            eng_tokenizer = load_tokenizer(self.config.eng_tokenizer_data_path)

            logger.info("Testing Translation Prediction")
            preds = np.argmax(model.predict(X_test), axis=-1)

            preds_text = []
            for i in preds:
                temp = []
                for j in range(len(i)):
                    t = get_word(i[j], eng_tokenizer)
                    if j > 0:
                        if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                            temp.append('')
                        else:
                            temp.append(t)
                    else:
                        if(t == None):
                            temp.append('')
                        else:
                            temp.append(t) 
            
                preds_text.append(' '.join(temp))

            pred_fra2eng = pd.DataFrame({'actual_fra' : test[:,1], 'actual_eng' : test[:,0], 'predicted_eng' : preds_text})

            pred_fra2eng.to_csv(self.config.results_data_path,index=False,header=True)

            print(pred_fra2eng.head(10))

            return pred_fra2eng
        
        except Exception as e:
            raise CustomException(e,sys)     
