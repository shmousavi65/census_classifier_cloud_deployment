# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
 - **Developer**: Seyed Hossein Mousavi
 - **Date**: 2023-11-17
 - **Version**: 0.0 (initial version)
 - **Type**: Logistic Regression Classifier

## Intended Use
  The developed pipeline classifies the salary group a person (binary: <=50K 0r >50K) based on the provided features named in the 'Training Data' section. This pipeline has been developed following the MLOPs best practices.   

## Training Data
<a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">Census Income</a> dataset has been used for both training and evaluation. 80% of the mentioned dataset has been selected randomly and used for training. The categorical features and numerical features are preprocessed by `One Hot Encoder` and `Standard Scaler` before feeding into the `Logistic Regression` model. Here are the list of used feature in the pipeline:
  - *Categorical features*: `workclass, education, marital-status, occupation, relationship, race, sex, native-country`.
  - *Numerical features*: `age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week`.

## Evaluation Data
20% of the <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">Census Income</a> dataset has been selected randomly and used for evaluation. all the preprocessing steps and used features are explained in the `Training Data` section. 

## Metrics
Here are the performance metrics and the values on the Evaluation data:
    - Precision: 0.74
    - Recall: 0.61
    - Fbeta: 0.67

## Ethical Considerations
This model has not been thoroughly investigated and has been only developed for learning purposes.

## Caveats and Recommendations
The model performance is not very good, as only the logistic regression model has been considered. It is strongly recommended to apply other classifiers like random-forest and compare their performances against current model.
