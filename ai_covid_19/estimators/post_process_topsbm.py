from ai_covid_19.hSBM_Topicmodel.sbmtm import sbmtm
import pandas as pd 
import numpy as np


def post_process_model(model,top_level,cl_level,top_thres=1):
    '''Function to post-process the outputs of a hierarchical topic model
      _____
      Args:
        model:      A hsbm topic model
        top_level:  The level of resolution at which we want to extract topics
        cl_level:   The level of resolution at which we want to extract clusters
        top_thres:  The maximum share of documents where a topic appears. 
                    1 means that all topics are included
      _____
      Returns:
        A topic mix df with topics and weights by document
        A lookup between ids and clusters
    '''
    #Extract the word mix (word components of each topic)
    word_mix = model.topics(l=top_level)

    #Create tidier names
    topic_name_lookup = {key:'_'.join([x[0] for x in values[:5]]
                                      ) for key,values in word_mix.items()}
    topic_names = list(topic_name_lookup.values())

    #Extract the topic mix df
    topic_mix_ = pd.DataFrame(model.get_groups(l=top_level)['p_tw_d'].T,
                            columns=topic_names,index=model.documents)

    #Remove highly uninformative / generic topics
    topic_prevalence = topic_mix_.applymap(lambda x: x>0
                                           ).mean().sort_values(ascending=False)
    filter_topics = topic_prevalence.index[topic_prevalence<top_thres]
    topic_mix = topic_mix_[filter_topics]

    #Extract the clusters to which different documents belong (we force all documents 
    #to belong to a cluster)
    cluster_assigment = model.clusters(l=cl_level,n=len(model.documents))
    cluster_sets = {c:set([x[0] for x in papers]) for c,papers in cluster_assigment.items()}

    #Assign topics to their clusters
    topic_mix['cluster'] = [
    [f'cluster_{n}' for n,v in cluster_sets.items() if x in v][0] for x in topic_mix.index]

    return topic_mix, topic_mix['cluster'].to_dict()
