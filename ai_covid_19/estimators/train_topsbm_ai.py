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

#Read data
rxiv_ai = pd.read_csv(f"{DATA_PATH}/rxiv_papers_update.csv",dtype={
                      'id':str,'is_ai':bool,
                      'is_covid':bool}).query("is_ai == True")

#Focus on 2019 onwards
rxiv_ai_recent = rxiv_ai.loc[rxiv_ai['year']>=2019]

#Pre-process data
logging.info("Clean and tokenising")

abst = rxiv_ai_recent['abstract']
abst = [re.sub("\n"," ",x) for x in abst]

ct = CleanTokenize(abst)
ct.clean().bigram(threshold=20).bigram(threshold=20)

docs = ct.tokenised
_ids = list(rxiv_ai_recent['id'])

#Train model
model = sbmtm()

logging.info("Making graph")
model.make_graph(docs,documents=_ids)

logging.info("Fitting model")
model.fit()

#Save model
logging.info("Saving model")

with open(f"{project_dir}/models/top_sbm/top_sbm_ai.p",'wb') as outfile:
    pickle.dump(model,outfile)