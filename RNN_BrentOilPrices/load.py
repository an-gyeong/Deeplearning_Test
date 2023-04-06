import pandas as pd

__all__=['loader']

def loader(path:str, col :str = 'Date') -> pd.DataFrame :
    df = pd.read_csv(path, parse_dates=[col])
    return df