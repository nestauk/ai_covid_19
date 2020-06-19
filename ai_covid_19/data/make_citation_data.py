import logging
import sys
import os
import ratelim
from dotenv import load_dotenv,find_dotenv
import json
import itertools
from collections import Counter, defaultdict
import numpy as np
import scipy as sp
import pandas as pd

import cord19
from nesta.packages.mag.query_mag_api import build_expr
from nesta.packages.mag.query_mag_api import query_mag_api

## Paths etc
PROJECT_PATH = cord19.project_dir
DATA_PATH = f'{project_dir}/data/processed'

load_dotenv(find_dotenv())
sql_config = os.getenv('config_path')
mag_key = os.getenv('mag_key')

logging.info(print(mag_key))

#####
#NOTE
####
#This script requres a Microsoft Academic Knowledge API key

#Read covid data
logging.info("Reading data")
xiv = pd.read_csv(f"{DATA_PATH}/ai_research/rxiv_papers.csv")
cov = xiv.query("is_covid == 1")

#Get mag ids
mag_ids = [int(id) for id in cov.mag_id if not pd.isnull(id)]

# Get the citations
logging.info('Getting citation info')

result_cont = []
for expr in build_expr(mag_ids, 'Id'):
    result = query_mag_api(expr, fields=['Id', 'CitCon'], subscription_key=mag_key)
    result_cont.append(result)

all_results = list(itertools.chain(*[x['entities'] for x in result_cont]))

# Mapping of {citing article id --> [list of citation article ids]}
citers = {int(article['Id']): list(article['CitCon'].keys()) 
          if 'CitCon' in article else [] for article in all_results}

# Set of ids of all cited articles
citee_ids = set(int(id) for id in itertools.chain.from_iterable(citers.values()))

logging.info(f"Number of unique citees: {len(citee_ids)}")

# Get full info for each citation
logging.info("getting citations")

results = []
query_count = 1000
for expr in build_expr(citee_ids.union(citers), 'Id'):
    count, offset = query_count, 0
    # Do until no results left
    while count == query_count:
        _result = query_mag_api(expr, fields=['Id', 'J.JN', 'D', 'DN', 'DOI', 'CC', 'F.FN'], 
                                subscription_key=mag_key, 
                                offset=offset, query_count=query_count)['entities']      
        count = len(_result)
        offset += count
        results += _result
        
# Data quality: check that we returned all of the citation IDs
returned_ids = {r['Id'] for r in results}
logging.info(print(len(citee_ids - returned_ids), len(set(citers) - returned_ids)))

# Look up for flattened variable names
field_dictionary = {'CC': 'citations', 
                    'D': 'date',
                    'DN': 'title',
                    'F': lambda x: {'fields_of_study': [_x['FN'] for _x in x]},
                    'Id': 'mag_id',
                    'J': lambda x: {'journal_title': x['JN']}}

# Mapping of all article ids (both citers and citees) --> flattened article data
articles = {}
for r in results:
    article = {}
    # Convert the field names from MAG to something legible
    for mag_key, field in field_dictionary.items():
        # Ignore this MAG field if the result doesn't have it!
        if mag_key not in r:
            continue
        # If the mapping is str --> value
        if type(field) is str:
            article[field] = r[mag_key]
        # Otherwise assume that the mapping is a lambda function
        else:
            article.update(field(r[mag_key]))
    articles[r['Id']] = article

# Mapping of all article ids (both citers and citees) --> flattened article data
with open(f'{DATA_PATH}/ai_article_mag_info.json', 'w') as f:
    f.write(json.dumps(articles))

# Citer ids. Together with `articles` you've got everything you need
with open(f'{DATA_PATH}/citation_lookup.json', 'w') as f:
    f.write(json.dumps(citers))

