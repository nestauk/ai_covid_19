import requests
import ai_covid_19
from zipfile import ZipFile
from io import BytesIO


project_dir = ai_covid_19.project_dir
FIGSH_PATH = 'https://ndownloader.figshare.com/files/'

def get_file(file):
    '''Download and extract a figshare file form our repo
    Args:
        fig (str) is the id for the file
    '''
    f = requests.get(f'{FIGSH_PATH}/{file}')
    z = ZipFile(BytesIO(f.content))
    return z





