"""
This script focuses cleaning and applying topic modeling (LDA)
articles extracted from https://www.uberpeople.net/forums/Complaints/
"""
# %% load libraries
# --+ basic
import os
import numpy as np
import pandas as pd
from pprint import pprint as pp
from datetime import datetime

# --+ data manipulation
import re
import spacy
import en_core_web_lg
import gensim
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# --+ model
from gensim.models import TfidfModel
import gensim.corpora as corpora
from gensim.models import LdaModel, ldamodel

# --+ basic visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --+ topic modeling visualizatioon
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --+ functions from other modules
from coherence_eval import compute_coherence_values

# --+ ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% set up MALLET mallet
mallet_path = '/Users/phuattagrish/mallet-2.0.8/bin/mallet'

# %% check ipython location
sys.executable

# %% change working directory
os.getcwd()
# os.chdir('data')
os.chdir('../scripts')

# %% load the data
with open ('data.pickle', 'rb') as fp:
    dict_ = pickle.load(fp)
# --+ article
articles = dict_['articles']
# --+ timestamp
time_stamp_ = dict_['timestamp']
# --+ put the data into DataFrame
articles_to_use = [x[0] for x in articles]
df = pd.DataFrame({
    'articles':articles_to_use,
    'timestamp':time_stamp_
})
df.head()
df.head()

# %% basic cleaning
# --+ get only year/month/date
def to_date(string):
    date_time_obj = datetime.strptime(string, '%Y-%m-%d')
    return date_time_obj
# --+ apply the to_date function to the dataframe
df['timestamp'] = df['timestamp'].str[:10].apply(to_date)

# --+ priliminary clean the article
df.loc[:, 'articles'] = df['articles'].str.replace('\n', '')
df.head()
docs = [doc.strip().lower() for doc in df.articles]
docs = [re.sub(r'\b-\b', '_', text) for text in docs]
len(docs)
type(docs)

# --+ export
_dict_ = {'df_nontokenized':df}
with open('df_nontokenized.pickle','wb') as f:
    pickle.dump(_dict_,f)

# %% advanced cleaning
# --+  load English spacy
nlp = spacy.load('en_core_web_lg')

# --+ clean the texts
docs_tokens, tmp_tokens = [], []
for doc in docs:
    tmp_tokens = [token.lemma_ for token in nlp(doc)
                  if not token.is_stop
                  and not token.is_punct
                  and not token.like_num
                  and token.is_alpha]
    docs_tokens.append(tmp_tokens)
    tmp_tokens = []

# bi-gram and tri-gram
# --+ get rid of common terms
common_terms = [u'of', u'with', u'without', u'and', u'or', u'the', u'a',
                u'not', u'be', u'to', u'this', u'who', u'in']
bigram = Phrases(docs_tokens,
                 min_count=10,
                 threshold=5,
                 max_vocab_size=50000,
                 common_terms=common_terms)
trigram = Phrases(bigram[docs_tokens],
                  min_count=10,
                  threshold=5,
                  max_vocab_size=50000,
                  common_terms=common_terms)

# --+ get tri-gram of tokenized words
docs_phrased = [trigram[bigram[line]] for line in docs_tokens]
DICT = Dictionary(docs_phrased)
CORPUS = [DICT.doc2bow(doc) for doc in docs_phrased]

# remove addtional noise words using TfidfModel
tfidf = TfidfModel(CORPUS, id2word=DICT)
# --+ set-up arguments
low_value = 0.03
words = []
words_missing_in_tfidf = []
# --+ loop over the CORPUS and remove noise words
for i in range(0, len(CORPUS)):
    bow = CORPUS[i]
    low_value_words = []
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(DICT[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
    new_bow = [b for b in bow if b[0] not in low_value_words and b[0]
               not in words_missing_in_tfidf]
    CORPUS[i] = new_bow

# %% export data
# --+ create a dataframe
df_to_export = pd.DataFrame({
    'articles':docs_phrased,
    'timestamp':df['timestamp']
})
df_to_export
# --+ export the data
os.getcwd()
os.chdir('../data')
dict_ = {'df_tokenized':df_to_export}
with open('df_tokenized.pickle','wb') as f:
    pickle.dump(dict_,f)

# %% compute coherance scores produced by several number of num_topics
# --+ set-up
limit, start, step = 10, 2, 1
tic = time.time()
model_list, coher_vals = compute_coherence_values(dictionary=DICT,
                                                  corpus=CORPUS,
                                                  texts=docs_phrased,
                                                  start= start,
                                                  limit= limit,
                                                  step= step)
toc = time.time()
print(toc - tic)
model_list
coher_vals

# --+ model with the optimal number of num_topics based on coherance value and interpretibility
lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path= '/Users/phuattagrish/mallet-2.0.8/bin/mallet',
                                              corpus=CORPUS,
                                              num_topics= 4,
                                              id2word=DICT,
                                              random_seed=123)

# --+ words associated with the optimal model
lda_mallet.print_topics(num_topics= 4,
                        num_words=10)


# %% result analyses
# interpret the result
# --+ get document-topic pairs probabilities
LDA_MALLET_G = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)
LDA_MALLET_G
TRANSF_CORPUS = LDA_MALLET_G.get_document_topics(CORPUS)
TRANSF_CORPUS
DOC_TOPIC_M = []
len(CORPUS)
for id, doc in enumerate(TRANSF_CORPUS):
    for topic in np.arange(0, 4, 1):
        topic_n = doc[topic][0]
        topic_prob = doc[topic][1]
        DOC_TOPIC_M.append([id, topic, topic_prob])
np.arange(0, 4, 1)

# --+ populate the dataframe
DF = pd.DataFrame(DOC_TOPIC_M)
DF # --> correct
DF.iloc[100:120]

# --+ rename columns
OLD_NAMES = [0, 1, 2]
NEW_NAMES = ['doc_id', 'topic_n', 'prob']
COLS = dict(zip(OLD_NAMES, NEW_NAMES))
DF.rename(columns=COLS, inplace=True)

# --+ get dominant topic
GR = DF.groupby('doc_id')
DF.loc[:, 'max'] = GR['prob'].transform(np.max)
DF
DF.loc[:, 'first_topic'] = 0
DF.loc[DF['prob'] == DF['max'], 'first_topic'] = 1
FIRST_TOPIC = DF.loc[DF['first_topic'] == 1]
FIRST_TOPIC
DF
# --+ drop the unused column
FIRST_TOPIC.drop(columns=['first_topic'], axis=1, inplace=True)
FIRST_TOPIC['topic_n'].value_counts()
FIRST_TOPIC
DF_TOPIC = FIRST_TOPIC.copy()
# --+ remove doc_id that cannot be assigned into a unique topic
DF_TOPIC = DF_TOPIC.loc[~FIRST_TOPIC.duplicated(subset=['doc_id'],keep=False), :]
# --+ export the data
os.getcwd()
dict_ = {'DF_TOPIC':DF_TOPIC}
with open('DF_TOPIC.pickle','wb') as f:
    pickle.dump(dict_,f)

# %% visualize number of each topic over the years
DF_TOPIC
# --+ to reduce granularity
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

# --+ manipulate data
df_vis =  DF_TOPIC.copy()
df_vis = df_vis.merge(df, how = 'inner', left_index = True, right_index = True)
df_vis['timestamp'] = df['timestamp'].astype(str).str[:7].apply(Quarter)
df_vis = df_vis[['topic_n','timestamp']]
df_vis['count'] = 1
df_vis = df_vis.groupby(['topic_n','timestamp']).agg(np.sum).reset_index()
df_vis.head()
unique_date = df_vis['timestamp'].unique().tolist()
for d in unique_date:
    topic_available = df_vis.loc[df_vis['timestamp']==d,'topic_n'].unique().tolist()
    for i in range(1,5):
        if i not in topic_available:
            df_concat = pd.DataFrame({
                        'topic_n':[i],
                        'timestamp':[d],
                        'count':[0]
                        })
            df_vis = pd.concat([df_vis,df_concat],axis=0)

df_vis = df_vis.reset_index(drop=True)
df_vis['X_label'] = 0

for i in range(1,5):
    for idx, d in enumerate(unique_date):
        df_vis.loc[(df_vis['topic_n'] ==i) & (df_vis['timestamp'] ==d),'X_label'] = idx

# --+ visualize
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(30,15))
x_labels = df_vis['timestamp'].unique().tolist()[1:-1]
labels = df_vis['topic_n'].unique().tolist()

t_1 = np.array(df_vis.loc[df_vis['topic_n']==0,'count'])[1:-1]
t_2 = np.array(df_vis.loc[df_vis['topic_n']==1,'count'])[1:-1]
t_3 = np.array(df_vis.loc[df_vis['topic_n']==2,'count'])[1:-1]
t_4 = np.array(df_vis.loc[df_vis['topic_n']==3,'count'])[1:-1]

ax.bar(x_labels,t_1,label=labels[0])
ax.bar(x_labels,t_2,label=labels[1],bottom=t_1)
ax.bar(x_labels,t_3,label=labels[2],bottom=t_2+t_1)
ax.bar(x_labels,t_4,label=labels[3],bottom=t_3+t_2+t_1)
ax.legend(fontsize=20,labels=['Topic '+str(x) for x in range(0,4)])

# --+ decorate
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.xlabel("Time")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.show()
# --+ save the figure
os.getcwd()
out_f = os.path.join("..","output", "topic_modeling.pdf")
plt.savefig(out_f, transparent=True, bbox_inches="tight", pad_inches=0)

# %% project unseen dataset to the model scraped from recent posts in complaint community
test_1 = ["Scruber emailed me and sent msg thru app on Friday Juky 2nd; $100 for 3 completed rides over 4th of July weekend. Figured do 3 quick trips and then done and go home w $100 + fares. Scruber paid the fares but not the promo. I sent msg & they replied ‘due to some outages, we are experiencing delays in response times & prioritizing emergencies.... That was 2 days ago & still no response despite repeated msgs to them. Scruber strikes again."]
test_2 = ["Anyone else get this I'm in Chicago. No way I am waiting 7 minutes for a pax. Since Uber has not been paying for the first 2 minutes after arrival, I am only waiting those 2 minutes from this point forward, then leaving. Uber knows that riders are waiting longer for drivers than ever before; now they are going to give them 7 MORE minutes after we arrive?? Good luck with that Uber and pax."]
test_all = test_1 + test_2

# --+ clean the unseen dataset
docs_tokens_test, tmp_tokens_test = [], []
for doc in test_all:
    tmp_tokens_test = [token.lemma_ for token in nlp(doc)
                  if not token.is_stop
                  and not token.is_punct
                  and not token.like_num
                  and token.is_alpha]
    docs_tokens_test.append(tmp_tokens_test)
    tmp_tokens_test = []

docs_tokens_test
# bi-gram and tri-gram
# --+ get rid of common terms
bigram = Phrases(docs_tokens_test,
                 min_count=10,
                 threshold=5,
                 max_vocab_size=50000,
                 common_terms=common_terms)
trigram = Phrases(bigram[docs_tokens_test],
                  min_count=10,
                  threshold=5,
                  max_vocab_size=50000,
                  common_terms=common_terms)

# --+ get tri-gram of tokenized words
docs_phrased_test = [trigram[bigram[line]] for line in docs_tokens_test]
DICT_test = Dictionary(docs_phrased_test)
CORPUS_test = [DICT_test.doc2bow(doc) for doc in docs_phrased_test]

# remove addtional noise words using TfidfModel
tfidf = TfidfModel(CORPUS_test, id2word=DICT)
# --+ set-up arguments
low_value = 0.03
words = []
words_missing_in_tfidf = []
# --+ loop over the CORPUS and remove noise words
for i in range(0, len(CORPUS_test)):
    bow = CORPUS_test[i]
    low_value_words = []
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(DICT[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
    new_bow = [b for b in bow if b[0] not in low_value_words and b[0]
               not in words_missing_in_tfidf]
    CORPUS_test[i] = new_bow
other_corpus = [DICT.doc2bow(text) for text in docs_phrased_test]

# %% anaylse the unseen documents and their results
vector_0 = lda_mallet[other_corpus[0]]
vector_1 = lda_mallet[other_corpus[1]]
vector_0
vector_1
