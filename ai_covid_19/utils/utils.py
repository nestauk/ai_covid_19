from toolz.curried import *

def preview(df):
    '''Previews a dataframe
    Args:
        x (df) a dataframe

    '''
    print(df.head())
    return df