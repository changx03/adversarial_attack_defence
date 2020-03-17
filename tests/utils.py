import os

def get_data_path():
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, 'data')
    return path