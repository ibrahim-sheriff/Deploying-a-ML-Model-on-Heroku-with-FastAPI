import os


def test_correct_data():
    assert os.listdir('../data')[0] == 'census.csv'
