import itertools 
import pandas as pd
from fastapi.testclient import TestClient
from pipeline.ml.model import load_model, inference

from main import app
from main import InputItem
# parameters
path_to_model = "model/model.pkl"
path_to_example_df = "data/census.csv"
example_df_index = 0

# load an example input
input_pipe, output_transformer = load_model(path_to_model)
used_columns = list(itertools.chain.from_iterable([x[2] for x in input_pipe['preprocessor'].transformers]))
used_columns_set = set(used_columns)

example_df = pd.read_csv(path_to_example_df)
example_series = example_df.iloc[example_df_index]
example_series = example_series[used_columns]
example_dict = example_series.to_dict()

client = TestClient(app)

def test_api_locally_do_greeting():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello User!"

def test_api_locally_post():
    input_dict = {"element": example_dict}
    r = client.post("/inference/", json=input_dict)
    
    out = r.json()
    assert r.status_code == 200 # assert response code is correct
    assert out["output"] == 0 # assert inference result is correct

def test_api_locally_post_wrong_input():
    example_dict.pop("age")
    input_dict = {"element": example_dict}
    r = client.post("/inference/", json=input_dict)
    assert r.status_code == 191 # assert response code is correct