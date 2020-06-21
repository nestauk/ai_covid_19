import numpy as np
import re
import random
from toolz.curried import *

def preview(df):
    '''Previews a dataframe
    Args:
        x (df) a dataframe

    '''
    print(df.head())
    return df

def convert_var(var,lookup):
    
    return(var.map(lookup))

def convert_covid(var):
    out = convert_var(var,{True:'COVID-19',False:'Not COVID-19'})
    return(out)
    
def convert_ai(var):
    
    out= convert_var(var,{True:'AI',False:'Not AI'})
    return(out)

def convert_group(var):
    out = convert_var(var,{'all_arxiv':'All corpus',
                      'covid':'COVID-19','ai':'AI','covid_ai':'AI and COVID-19'})
    return(out)


def convert_source(var):
    out = convert_var(var, {'medrxiv':'medrXiv','arxiv':'arXiv','biorxiv':'biorXiv'})
    
    return(out)

def make_pc(var,scale=1):
    return([str(np.round(scale*x,2))+'%' for x in var])

def clean_cluster(cluster_var):
    return([re.sub('_',' ',x.capitalize()) for x in cluster_var])

def clean_topics(topic_var):
    
    return([', '.join([re.sub(
        '-','',x.capitalize()) for x in mix.split('_')]) for mix in topic_var])

def get_examples(_list,values):
    if len(_list)==0:
        return('')
    elif len(_list)<values:
        return(random.choices(_list,k=len(_list)))
    else:
        return(random.choices(_list,k=values))