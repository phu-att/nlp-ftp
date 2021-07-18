"""
This script implements the coherance score
that statistically evaluates number of topics
"""
# %% import relevant libraries
# %% import relevant libraries
# basic
import os
import numpy as np
import pandas as pd

# data manipulation
import re
import spacy
import en_core_web_lg
import gensim
from gensim.models import Phrases
from gensim.models import LdaModel, ldamodel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


# model
from gensim.models import TfidfModel
import gensim.corpora as corpora

# set up MALLET_PATH
MALLET_PATH = '/Users/phuattagrish/mallet-2.0.8/bin/mallet'
# %% create lda evaluator metric
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    -----------
    dictionary : Gensim dictionary
    corpus     : Gensim corpus
    texts      : List of input texts
    limit      : Max number of topics

    Returns:
    --------
    model_list       : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model
                       with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = MALLET_PATH
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                 corpus=corpus,
                                                 num_topics=num_topics,
                                                 id2word=dictionary,
                                                 random_seed=123)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
