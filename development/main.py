import itertools 
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from pipeline.ml.model import load_model, inference

# parameters
path_to_model = "model/model.pkl"
path_to_example_df = "data/census.csv"
example_df_index = 0

# load the model
input_pipe, output_transformer = load_model(path_to_model)
used_columns = list(itertools.chain.from_iterable([x[2] for x in input_pipe['preprocessor'].transformers]))
used_columns_set = set(used_columns)

#load an example
example_df = pd.read_csv(path_to_example_df)
example_series = example_df.iloc[example_df_index]
example_series = example_series[used_columns]
example_dict = example_series.to_dict()

class InputItem(BaseModel):
    element : dict

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
        raise HTTPException(status_code=191, detail="The input does not contain the required values!")
    
    input_df = pd.DataFrame.from_dict(input.element, orient='index').transpose()
    if len(input_df) != 1:
        raise HTTPException(status_code=193, detail="The input does should only contain 1 sample!")
    
    output = inference(input_pipe, input_df)
    return {"output": int(output[0])}