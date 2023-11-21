import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--data_path", action="store")


@pytest.fixture(scope="session")
def data(request):
    data_path = request.config.option.data_path
    df = pd.read_csv(data_path)
    return df
