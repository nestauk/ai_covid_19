#This script creates the main dataset
#It requires access to Nesta's data production system and its data_getters
#If you don't have access to this, follow the instructions in the README

import logging
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv,find_dotenv
from data_getters.core import get_engine
from ai_covid_19.transformers.nlp import clean_and_tokenize
from gensim.models.phrases import Phrases, Phraser
import ai_covid_19

#Directories etc
project_dir = ai_covid_19.project_dir
DATA_PATH  = os.path.join(project_dir,'data/processed')

load_dotenv(find_dotenv())

config_path = os.getenv('config_path')

#Metadata
covid_keywords = ai_covid_19.config["keywords"]["covid_19"]
ml_keywords = ai_covid_19.config["keywords"]["ai"]

#####
#Read rXiv data
#####

#Collects the data
con = get_engine(f"{config_path}")
logging.info("Downloading data")

chunks = pd.read_sql_table("arxiv_articles", con, chunksize=1000)
papers = pd.concat(chunks)

logging.info(len(papers))

#Processing and new variables
papers = papers.reset_index(drop=True)
papers = papers.dropna(subset=["title", "abstract"])
papers["year"] = papers.created.apply(lambda x: x.year)

########
# Label covid-19 papers
########

logging.info("Finding Covid papers")

papers["is_covid"] = [
    1
    if any(term.lower() in row["abstract"].lower() for term in covid_keywords)
    or any(term.lower() in row["title"].lower() for term in covid_keywords)
    else 0
    for idx, row in papers.iterrows()
]

logging.info(papers['is_covid'].sum())

#########
# Label AI papers
#########
logging.info("Finding AI papers")

# Text pre-proecessing
abstracts = [clean_and_tokenize(d, remove_stops=True) for d in papers.abstract]
phrases = Phrases(abstracts, min_count=5, threshold=10)
bigram = Phraser(phrases)
trigram = Phrases(bigram[abstracts], min_count=5, threshold=3)
abstracts_with_ngrams = list(trigram[abstracts])

#The keywords were identified through a previous expanded search

papers["is_ai"] = [
    1 if any(k in tokens for k in ml_keywords) else 0
    for tokens in abstracts_with_ngrams
]

logging.info(papers['is_ai'].sum())

###########
# OUTPUTS
###########
logging.info("Saving outputs")

papers.to_csv(f"{DATA_PATH}/rxiv_papers_update.csv",index_label=False)