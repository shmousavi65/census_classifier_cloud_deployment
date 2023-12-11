The project follows production-level code practices to develop and deploy a classification model for income on Census data. The entire pipeline is developed as an MLOps process, with MLflow used to create isolated pipeline components and track experiments (including parameters, performance metrics, etc.). Data Version Control (DVC) is integrated with Amazon AWS and is used to store and track data and trained models. The project also uses GitHub Actions (for CI) and Heroku (for CD) to implement CI/CD practices and build an end-to-end, fully reproducible, and automated ML pipeline. It’s important to note that the primary goal of this project is to showcase MLOps skills, rather than to achieve superior model performance.

## Environment Set up
Note that WSL1 (Ubuntu-20.04) has been used for the whole processes.

creat a new python (version: 3.8.10) envrionment called `env` using the following:
```
python -m venv env
```
install the requirements by
```
pip install -r requirements.txt
```

## Setup DVC

DVC is used to track the trained`model`s and the `data`. `init dvc` in the cloned repo. An AWS S3 bucket is created and set as remote for dvc.

### Set up S3

* In your CLI environment install the<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html" target="_blank"> AWS CLI tool</a>.
* login to aws account
* From the Services drop down select S3 and then click Create bucket.
* Give your bucket a name, the rest of the options can remain at their default.

* To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. The full instructions can be found <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console" target="_blank">here</a>, what follows is a paraphrasing:

* Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
* In the left navigation bar select **Users**, then choose **Add user**.
* Give the user a name and select **Programmatic access**.
* In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
* Tags are optional and can be skipped.
* After reviewing your choices, click create user. 
* Configure your AWS CLI to use the Access key ID and Secret Access key.

Once the above steps are done you can add the s3 as the default remote:

```
dvc remote add --default myremote s3://your-bucket-name-here
```

## Data
* census.csv is downloaded from <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a> and is cleaned (all unnecessary spaces are removed). The data is stored in `development/data`.
   
* track the data using dvc 
    ```
    dvc add data
    ```
    and then push it to the remote (S3 bucket)
    ```
    dvc push
    ```
    the samoe steps can be used to store and track the saved trained models. The trained model is stores in `development/model`. 

## MLOps Development and Experiment Tracking by MLflow
MLflow is used to develop, The full pipeline consists of the following component:
- data_check: verify some statistical and deterministic characteristics of input data through tests. 
- split_data: split the data to train/test portions.
- model_train: preprocess features and train the model 
- performace_eval: evaluate of the trained model performance on the test data 

Also, experiment (including parameters, performance metrics, ...) are tracked by mlflow.

## Continuous Integration (CI) using GitHub Actions

* GitHub Actions has bee set on the repo.
* specifically it runs the following actions on the push and pull request:
* run flake8
* configure AWS credentials for dvc to be able to pull the saved models (and possibly data). usefule link: <a href="https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions" target="_blank">AWS credentials to the Action</a>. (Note: for this step AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY has been in as the Repository secrets of the repo, in the gthub.) 
* pull the saved models (and possibly data) using dvc (the model is required during testing with pytest). useful link: <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a>.
* run test using pytest

## RESTful API Development using FastAPI
The script for the FastAPI development can be found in `development/main.py`. This API includes GET and POST methods: 
  * GET on the root giving a welcome message.
  * POST that does model inference.

In order to run the API, run the following in root directory:
```
uvicorn development.main:app
```
which runs the API on the local server http://127.0.0.1:8000. In order to see the help docs on the API and also examples on how to run do inference by the model, go to: http://127.0.0.1:8000/docs. 


## Continuous Delivery (CD) using Deployment on Heroku

The following steps have been done to create an app on Heroku and successfuly link the github repo with it to accomplish the continuous delivery:

* Create a new app and have it deployed from your GitHub repository.
* Enable automatic deployments that only deploy if your continuous integration passes.
* Set up DVC on Heroku using the instructions contained in the `dvc_on_heroku_instructions`.
* Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy` (Note: this would save Config Vars, which can be found in the `setting` of the related app in your Heroku account)

Once the app is running, you can run the following command in the root directory to do a sample POST on the live API:
```
python live_app_requests.py
```
### Running Pipeline using MLflow
In order to run the pipeline.
- change the working directory to `development/pipeline`,
- if required, change the pipeline parameters found in `development/pipeline/config.yaml`. These parameters include name of experiment, desired pipeline componenets to implement, train/test parameters, etc. 
- run the following command:
  ```
  python pipeline.py
  ``` 

### Performance Evaluation using MLflow
by running the following command in the `development/pipeline/outputs` directory you can get a web-based representation of experiment informations tracked by mlflow:
```
mlflow ui
```

### Credit
This project has been done as a part of Udacity's ML DevOps Engineer nanodegree program. You can follow and find all the changes that this repo has done compared to the original starter repo (which this repo is forked from). 

