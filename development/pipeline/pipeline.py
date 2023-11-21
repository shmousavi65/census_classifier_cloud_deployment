import mlflow
import os
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration


@hydra.main(config_name='config')
def go(config: DictConfig):

    mlflow.set_experiment(config["experiment_name"])

    root_path = hydra.utils.get_original_cwd()
    data_model_parent_dir = os.path.dirname(os.path.dirname(root_path))

    exec_steps_list = list(config["execution_steps"])

    if "data_check" in exec_steps_list:
        _ = mlflow.run(
            os.path.join(
                root_path,
                "data_check"),
            "main",
            parameters={
                "data_path": os.path.join(
                    data_model_parent_dir,
                    config["data"]["raw_data_path"])},
            env_manager="local")

    if "split_data" in exec_steps_list:
        _ = mlflow.run(
            os.path.join(
                root_path,
                "split_data"),
            "main",
            parameters={
                'raw_data_path': os.path.join(
                    data_model_parent_dir,
                    config["data"]["raw_data_path"]),
                'random_state': config['train']['random_state'],
                'test_size': config['train']['test_size'],
            },
            env_manager="local")

    if "model_train" in exec_steps_list:
        raw_data_path = os.path.join(
            data_model_parent_dir,
            config["data"]["raw_data_path"])
        raw_dirname = os.path.dirname(raw_data_path)
        raw_basename = os.path.basename(raw_data_path)
        train_basename = raw_basename.split(".")[0] + "_train.csv"
        train_data_path = os.path.join(raw_dirname, train_basename)
        _ = mlflow.run(
            os.path.join(
                root_path,
                "model_train"),
            "main",
            parameters={
                'data_path': train_data_path,
                "numerical_features": config['train']['num_features'],
                "categorical_features": config['train']['cat_features'],
                "model_params": config['train']['model_params'],
                'model_save_path': os.path.join(
                    data_model_parent_dir,
                    config["model"]["model_save_path"]),
                'output_label': config['train']['output_label']},
            env_manager="local")

    if "performace_eval" in exec_steps_list:
        raw_data_path = os.path.join(
            data_model_parent_dir,
            config["data"]["raw_data_path"])
        raw_dirname = os.path.dirname(raw_data_path)
        raw_basename = os.path.basename(raw_data_path)
        test_basename = raw_basename.split(".")[0] + "_test.csv"
        test_data_path = os.path.join(raw_dirname, test_basename)
        _ = mlflow.run(
            os.path.join(
                root_path,
                "performance_eval"),
            "main",
            parameters={
                'data_path': test_data_path,
                'model_path': os.path.join(
                    data_model_parent_dir,
                    config["model"]["model_save_path"]),
                'output_label': config['train']['output_label'],
                'slice_eval_features':
                    config['evaluation']['slice_eval_features']},
            env_manager="local")


if __name__ == "__main__":
    go()
