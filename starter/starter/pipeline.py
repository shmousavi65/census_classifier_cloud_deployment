import mlflow
import os
import hydra
import ast
from omegaconf import DictConfig, OmegaConf

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    root_path = hydra.utils.get_original_cwd()

    exec_steps_list = list(config["execution_steps"])

    if "data_check" in exec_steps_list:
        _ = mlflow.run(
                os.path.join(root_path, "data_check"),
                "main",
                parameters={
                    "data_path": config["data"]["raw_data_path"]
                },
                env_manager="local"
            )

    if "split_data" in exec_steps_list:
        _ = mlflow.run(
                os.path.join(root_path, "split_data"),
                "main",
                parameters={
                    'raw_data_path' : config["data"]["raw_data_path"],
                    'random_state' : config['train']['random_state'],
                    'test_size' :  config['train']['test_size'],
                },
                env_manager="local"
            )
    
    if "model_train" in exec_steps_list:
        raw_data_path = config["data"]["raw_data_path"]
        raw_dirname = os.path.dirname(raw_data_path)
        raw_basename = os.path.basename(raw_data_path)
        train_basename = raw_basename.split(".")[0] + "_train.csv"  
        train_data_path = os.path.join(raw_dirname, train_basename)
        _ = mlflow.run(
                os.path.join(root_path, "model_train"),
                "main", 
                parameters={
                    'data_path' : train_data_path,
                    "numerical_features": config['train']['num_features'],
                    "categorical_features": config['train']['cat_features'],
                    "model_params":config['train']['model_params'],
                    'model_save_path': config['model']['model_save_path'],
                    'output_label': config['train']['output_label']
                },
                env_manager="local"
            )

    if "performace_eval" in exec_steps_list:
        raw_data_path = config["data"]["raw_data_path"]
        raw_dirname = os.path.dirname(raw_data_path)
        raw_basename = os.path.basename(raw_data_path)
        test_basename = raw_basename.split(".")[0] + "_test.csv"  
        test_data_path = os.path.join(raw_dirname, test_basename)
        _ = mlflow.run(
                os.path.join(root_path, "performance_eval"),
                "main", 
                parameters={
                    'data_path' : test_data_path,
                    'model_path': config['model']['model_save_path'],
                    'output_label': config['train']['output_label'],
                    'slice_eval_features': config['evaluation']['slice_eval_features']
                },
                env_manager="local"
            )

if __name__ == "__main__":
    go()

