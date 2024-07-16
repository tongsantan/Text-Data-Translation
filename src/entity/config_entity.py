## 3. Update the entity

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    input_data_path: Path
    word_array_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    word_array_path: Path
    eng_tokenizer_data_path: Path
    fra_tokenizer_data_path: Path
    test_array_path: Path
    X_train_array_path: Path
    y_train_array_path: Path
    X_test_array_path: Path
    y_test_array_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    X_train_array_path: Path
    y_train_array_path: Path
    eng_tokenizer_data_path: Path
    fra_tokenizer_data_path: Path
    trained_model_file_path: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_data_path: Path
    test_array_path: Path
    X_test_array_path: Path
    eng_tokenizer_data_path: Path
    results_data_path: Path