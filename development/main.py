import itertools
import pandas as pd
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
sys.path.append('./pipeline/')
from ml.model import load_model, inference

# parameters
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_model = os.path.join(current_dir,"model/model.pkl")
path_to_example_df = os.path.join(current_dir,"data/census.csv")
example_df_index = 0

# load the model
input_pipe, output_transformer = load_model(path_to_model)
used_columns = list(itertools.chain.from_iterable(
    [x[2] for x in input_pipe['preprocessor'].transformers]))
used_columns_set = set(used_columns)

# load an example
example_df = pd.read_csv(path_to_example_df)
example_series = example_df.iloc[example_df_index]
example_series = example_series[used_columns]
example_dict = example_series.to_dict()


class InputItem(BaseModel):
    element: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                example_dict
            ]
        }
    }


# create app instance
app = FastAPI()


@app.get("/")
async def do_greeting():
    return "Hello User!"


@app.post("/inference/")
async def do_inference(input: InputItem):
    input_keys = set(input.element.keys())
    if used_columns_set.difference(input_keys):
        raise HTTPException(
            status_code=191,
            detail="The input does not contain the required values!")

    input_df = pd.DataFrame.from_dict(
        input.element, orient='index').transpose()
    if len(input_df) != 1:
        raise HTTPException(
            status_code=193,
            detail="The input does should only contain 1 sample!")

    output = inference(input_pipe, input_df)
    return {"output": int(output[0])}
