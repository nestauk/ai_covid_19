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

import cord19
from cord19.hSBM_Topicmodel.sbmtm import sbmtm
from cord19.transformers.nlp import *

## Paths
project_dir = cord19.project_dir
DATA_PATH = f"{project_dir}/data/processed/"

#####
#Read and process data
#####
logging.info("Reading data")

xiv = pd.read_csv(f"{DATA_PATH}/xiv_papers.csv",
                  dtype={'id':str})
cov_ = xiv.query('is_covid == 1').reset_index(drop=False).pipe(preview)

#Extract AI ids
ai_ids = set(cov_.query('is_ai == 1')['id'])
logging.info(print(len(ai_ids)))

#Drop papers without abstracts and with ver
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

#Extract the word mix (word components of each topic)
word_mix = model.topics(l=0)

#Create tidier names
topic_name_lookup = {key:'_'.join([x[0] for x in values[:5]]
                                  ) for key,values in word_mix.items()}
topic_names = list(topic_name_lookup.values())

#Extract the topic mix df
topic_mix_ = pd.DataFrame(model.get_groups(l=0)['p_tw_d'].T,
                        columns=topic_names,index=list(cov['id']))

#Remove highly uninformative / generic topics
topic_prevalence = topic_mix_.applymap(lambda x: x>0
                                       ).mean().sort_values(ascending=False)
topic_prevalence.loc[topic_prevalence>0.4]
filter_topics = topic_prevalence.index[topic_prevalence<0.4]
topic_mix = topic_mix_[filter_topics]

#Extract the clusters to which different documents belong (we force all documents 
#to belong to a cluster)
cluster_assigment = model.clusters(l=1,n=len(list(cov['id'])))
cluster_sets = {c:set([x[0] for x in papers]) for c,papers in cluster_assigment.items()}

#Assign topics to their clusters
topic_mix['cluster'] = [
[f'cluster_{n}' for n,v in cluster_sets.items() if x in v][0] for x in topic_mix.index]

######
#Save outputs
######
logging.info("Saving data")
topic_mix_long = topic_mix.reset_index(drop=False
                                       ).melt(id_vars=['index','cluster']
                                       ).rename(columns={'index':'article_id'}
                                       )

with open(f"{DATA_PATH}/sbm_topic_mix.p",'wb') as outfile:
    pickle.dump(topic_mix_long,outfile)