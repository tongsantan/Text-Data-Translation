import os
from pathlib import Path
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, 
                                      DataTransformationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig)

## 4. Update the configuration manager in src config

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)

        create_directories([self.config.output_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            input_data_path=config.input_data_path,
            word_array_path=config.word_array_path
        )

        return data_ingestion_config 

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            word_array_path=config.word_array_path,
            eng_tokenizer_data_path=config.eng_tokenizer_data_path,
            fra_tokenizer_data_path=config.fra_tokenizer_data_path,
            test_array_path=config.test_array_path,
            X_train_array_path=config.X_train_array_path,
            y_train_array_path=config.y_train_array_path,
            X_test_array_path=config.X_test_array_path,
            y_test_array_path=config.y_test_array_path
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            X_train_array_path=config.X_train_array_path,
            y_train_array_path=config.y_train_array_path,
            eng_tokenizer_data_path=config.eng_tokenizer_data_path,
            fra_tokenizer_data_path=config.fra_tokenizer_data_path,
            trained_model_file_path=config.trained_model_file_path
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_data_path=config.model_data_path,
            test_array_path=config.test_array_path,
            X_test_array_path=config.X_test_array_path,
            eng_tokenizer_data_path=config.eng_tokenizer_data_path,
            results_data_path=config.results_data_path
        )

        return model_evaluation_config        