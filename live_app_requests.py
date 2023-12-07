import requests
import itertools
import sys
import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
# prev_dir = os.path.dirname(current_dir)
# sys.path.append(prev_dir)
sys.path.append(os.path.join(current_dir, "development", "pipeline"))
from ml.model import load_model

# parameters
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_model = os.path.join(current_dir, "development", "model/model.pkl")
path_to_example_df = os.path.join(current_dir,
                                  "development",
                                  "data/census.csv")
example_df_index = 0

# load an example input
input_pipe, output_transformer = load_model(path_to_model)
used_columns = list(itertools.chain.from_iterable(
    [x[2] for x in input_pipe['preprocessor'].transformers]))
used_columns_set = set(used_columns)

example_df = pd.read_csv(path_to_example_df)
example_series = example_df.iloc[example_df_index]
example_series = example_series[used_columns]
example_dict = example_series.to_dict()

app_get_url = "https://census-classifier-cloud-deploy-ebb5c04da69c." \
    "herokuapp.com/"
app_post_url = "https://census-classifier-cloud-deploy-ebb5c04da69c." \
    "herokuapp.com/inference/"


if __name__ == "__main__":
    input_dict = {"element": example_dict}
    r = requests.post(app_post_url, json=input_dict)
    out = r.json()
    print("output status code: ", r.status_code)
    print("inference result: ", out["output"])
