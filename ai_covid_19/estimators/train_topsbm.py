#This scripts trains the hieratchical topic model. 
#Note that this is available for reproducibility and future updates
#but will yield slightly different 
#results if rerun

import logging
import numpy as np
import scipy as sp
import pandas as pd
import gensim
import pickle
from toolz.curried import *

import ai_covid_19
from ai_covid_19.hSBM_Topicmodel.sbmtm import sbmtm
from ai_covid_19.transformers.nlp import *
from ai_covid_19.estimators.post_process_topsbm import *

## Paths
project_dir = ai_covid_19.project_dir
DATA_PATH = f"{project_dir}/data/processed"

#####
#Read and process data
#####
logging.info("Reading data")

xiv = pd.read_csv(f"{DATA_PATH}/rxiv_papers_update.csv",
                  dtype={'id':str})
cov_ = xiv.query('is_covid == 1').reset_index(drop=False)

#Extract AI ids
ai_ids = set(cov_.query('is_ai == 1')['id'])
logging.info(print(len(ai_ids)))

#Drop papers without abstracts and with very short abstracts
cov_ = cov_.dropna(axis=0,subset=['abstract'])
cov = cov_.loc[[len(x)>300 for x in cov_['abstract']]]
cov.reset_index(drop=True,inplace=True)

#Create a lookup between IDs and MAG_IDS
id_magid_lookup = {r['id']:r['mag_id'] for rid,r in cov.iterrows()}

#####
#Pre-process data
#####
logging.info("Clean and tokenising")

abst = cov['abstract']
abst = [re.sub("\n"," ",x) for x in abst]

ct = CleanTokenize(abst)

ct.clean().bigram(threshold=20).bigram(threshold=20)

docs = ct.tokenised
_ids = list(cov['id'])

logging.info("Training model")
model = sbmtm()
model.make_graph(docs,documents=_ids)
model.fit()

#Save model
logging.info("Saving model")

with open(f"{project_dir}/models/top_sbm/top_sbm.p",'wb') as outfile:
    pickle.dump(model,outfile)

######
#Post-process data
######
logging.info("Post-processing data")

#We use the previously defined function
post = post_process_model(model,top_level=0,cl_level=1,top_thres=0.4)

topics = post[0].reset_index(drop=False
                                       ).melt(id_vars=['index','cluster'],var_name='topic',value_name='weight'
                                       ).rename(columns={'index':'article_id'}
                                       )
topics['is_ai'] = topics['article_id'].isin(ai_ids)

######
#Save outputs
######
logging.info("Saving data")
topic_mix_long = topic_mix.reset_index(drop=False
                                       ).melt(id_vars=['index','cluster','is_ai']
                                       ).rename(columns={'index':'article_id'}
                                       )

with open(f"{DATA_PATH}/sbm_topic_mix.p",'wb') as outfile:
    pickle.dump(topic_mix_long,outfile)