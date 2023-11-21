import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_feature_presence(data):
    '''assert the features used for training are correct'''
    ref_features = {
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country'
    }

    data_features = set(data.columns)
    data_features.remove('salary')
    assert data_features == ref_features


def test_age_limit(data):
    '''assert the age is in the current range'''
    assert data.age.between(0, 122).all()


def test_workclass_classes(data):
    '''assert the classes in the workclasses feature'''
    assert set(data.workclass.unique()) == {' State-gov',
                                            ' Self-emp-not-inc',
                                            ' Private',
                                            ' Federal-gov',
                                            ' Local-gov',
                                            ' ?',
                                            ' Self-emp-inc',
                                            ' Without-pay',
                                            ' Never-worked'}
