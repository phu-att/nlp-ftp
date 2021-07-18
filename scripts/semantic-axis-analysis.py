"""
This script analyses affective states represented
in talks among Uber drivers using Semantic Axis Method (SAM)
and further incoporate topic modeling for time-series analysis
"""
# %% load libraries
import os
import numpy as np
from scipy import spatial
import pandas as pd
import spacy
import en_core_web_lg
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
# %% initialize a spaCy's pipeline
nlp = en_core_web_lg.load()

# %% load the data
os.getcwd()
# --+ harvard IV
df = pd.read_excel('data/inquireraugmented.xls', usecols=["Entry", "Hostile","Active" ,"Submit", 'Passive'])
df.head()
# --+ textual data

# %% get words associated with each column that is not null
hostile = df.loc[df["Hostile"].notnull(), "Entry"].to_list()[1:]
submit = df.loc[df["Submit"].notnull(), "Entry"].to_list()[1:]
active = df.loc[df["Active"].notnull(), "Entry"].to_list()[1:]
passive = df.loc[df["Passive"].notnull(), "Entry"].to_list()[1:]

# %% clean the words
# --+ lower words and remove part after #
def clean(word):
    return word.lower().split("#")[0]
# --+ apply the function
hostile = [clean(word) for word in hostile]
submit = [clean(word) for word in submit]
active = [clean(word) for word in active]
passive = [clean(word) for word in passive]
len(hostile)
hostile[0:10]
# %% get the affect score for a sample of unseen words
def semantic_ax(word_list, vector_len=300):
    wv = {}
    # step 2, get the word vectors
    for word in word_list:
        wv[word] = nlp.vocab[word].vector
    # step 3, get the centroid of each pole --> check understanding ##
    centroid = []
    for i in range(vector_len):
        dimension = [wv[word][i] for word in wv.keys()]
        centroid.extend([np.mean(dimension)])
    # return data
    return wv, np.array(centroid)


hostile_wv, hostile_centroid = semantic_ax(hostile)
submit_wv, submit_centroid = semantic_ax(submit)
active_wv, active_centroid = semantic_ax(active)
passive_wv, passive_centroid = semantic_ax(passive)

# %% step 4, get the semantic axis
my_ax = (hostile_centroid + active_centroid) - (submit_centroid + passive_centroid)
len(my_ax)
my_ax
# %% load the tokenized articles
os.getcwd()
with open ('data/df_tokenized.pickle', 'rb') as fp:
    dict_ = pickle.load(fp)
df_articles = dict_['df_tokenized']
df_articles

# %% project the articles to the pole we created
# --+ laod the untokenized data to be analysed
os.getcwd()
with open ('data/df_nontokenized.pickle', 'rb') as fp:
    dict_ = pickle.load(fp)
df_nontokenized = dict_['df_nontokenized']

# --+ loop over each row of the article to get the average cosine cosine_similarity between the pole and articles
cosine_ = []
for i in range(df_articles.shape[0]):
    articles_ = df_articles.iloc[i, 0]
    cosine_similarities = []
    for word in articles_:
        # consider words that have values in SpaCy and that are in the Harvard IV list we consider
        if nlp.vocab[word].vector is not None and word in hostile + active + passive + submit:
            cosine_similarity = 1- spatial.distance.cosine(my_ax, nlp.vocab[word].vector)
            cosine_similarities.append(cosine_similarity)
        else:
            pass
    cosine_.append(np.nanmean(cosine_similarities))
# --+ plot cosine histogram
rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
rc('text', usetex=True)
fig, ax = plt.subplots(figsize = (10,8))
ax.hist(cosine_)
ax.grid(True, ls='--')
ax.set_xlabel('Cosine Value')
ax.set_ylabel('Count')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
## --+ save plot
os.getcwd()
out_f = os.path.join("output", "cosine_hist.pdf")
plt.savefig(out_f, bbox_inches="tight", pad_inches=0)
plt.show()
# --+ get 10 highest cosine values
cosine_h = sorted(cosine_, reverse= True)[0:10]
cosine_h
cosine_.index(0.0013355350432296593)
# --+ get 10 indices associated with the consine values
to_inves_h = []
for i in set(cosine_h):
    to_inves.append(cosine_.index(i))
to_inves_h
# --+ get 10 lowest cosine values
cosine_l = sorted(cosine_)[0:10]
cosine_l

# --+ get 10 indices associated with the consine values
to_inves_l = []
for i in set(cosine_l):
    to_inves_l.append(cosine_.index(i))
to_inves_l

# %% check the df_nontokenized articles
# --+ 8 highest
for i in to_inves_h:
    print(df.iloc[i,0])
    print('/n')
# --+ 8 lowest
for i in to_inves_l:
    print(df.iloc[i,0])
    print('/n')

# %% analyse the sentimental changes through time
df_nontokenized.loc[:, 'urgency_score'] = cosine_
to_classify = pd.qcut(df_nontokenized['urgency_score'], 3, labels=["Non-Urgent", "Neutral", "Urgent"]) # might use other than qcut
df_nontokenized.loc[:, 'urgency_level'] = to_classify
df_nontokenized['urgency_level'].value_counts()
df_nontokenized

# %% merge with part 1 (topic modelling)
# --+ load the dataset
with open ('data/DF_TOPIC.pickle', 'rb') as fp:
    dict_ = pickle.load(fp)
DF_TOPIC = dict_['DF_TOPIC']
DF_TOPIC
DF_TOPIC.reset_index(drop=True, inplace=True)
DF_TOPIC.set_index('doc_id', inplace = True)
DF_TOPIC
# --+ merge two dataframes
df_combined = DF_TOPIC.merge(df_nontokenized, how = 'inner', left_index = True, right_index = True)

# %% visualize topic modeling and semantic changes over time
df_Q_ = df_combined.copy()
# --+ change date and month to quarter
def Quarter(string):
    month = int(string[5:])
    if month in [1,2,3]:
        string = string[:4] + '_Q1'
    if month in [4,5,6]:
        string = string[:4] + '_Q2'
    if month in [7,8,9]:
        string = string[:4] + '_Q3'
    if month in [10,11,12]:
        string = string[:4] + '_Q4'
    return string
# --+ manipulate data to make it visualizeable
df_Q_['timestamp'] = df_Q_['timestamp'].astype(str).str[:7].apply(Quarter)
df_Q_
df_Q_['timestamp']= df_Q_['timestamp'].astype(str).str[:4]
df_vis = pd.DataFrame(df_Q_.groupby(['timestamp', 'topic_n']).urgency_level.value_counts())
df_vis

# --+ visualize
rc('text', usetex=False)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize = (40,20))
df_vis.unstack().plot(kind='bar', stacked = True, ax = ax )
# --+ decorate
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.xlabel("(Time, Topic)")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)

# --+ write plot to file
os.getcwd()
out_f = os.path.join("output", "themes_affective_time_series.pdf")
plt.savefig(out_f, transparent=True, bbox_inches="tight", pad_inches=0)
plt.show()
# ax.bar(df_vis.unstack(), stacked = True)
# %% in 2021 Q1, financial issue is still an issue
