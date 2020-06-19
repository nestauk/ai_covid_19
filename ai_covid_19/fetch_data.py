import logging
import yaml
import ai_covid_19
import zipfile
from ai_covid_19.data.collect_figshare_data import get_file
import os

project_dir = ai_covid_19.project_dir
DATA_PATH = project_dir / 'data/processed'

# Model config
with open(project_dir / 'model_config.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())

files = config['fig_files']

print(files)

for f in files:
    logging.info(f)
    z = get_file(f)
    z.extractall(DATA_PATH)    




