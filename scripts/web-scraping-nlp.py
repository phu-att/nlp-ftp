"""
This script focuses on extracting texts
from articles in https://www.uberpeople.net/forums/Complaints/
and preliminary clean the data
"""
# %% load libraries
import requests
from bs4 import BeautifulSoup
import urllib.request
import sys
import time
import os
import datetime
import pickle

# %% check ipython location
sys.executable

# %% change working directory
os.getcwd()

# %% focus on the complaint community
|# --+ set-up
num_pages = 361
links = []
time_stamp = []

# %% get links of articles and its associated posted time to analyse
URL = 'https://www.uberpeople.net/forums/Complaints/'
tic = time.time()
for page in range(0,num_pages):
    if page == 0:
        URL_consider = 'https://www.uberpeople.net/forums/Complaints/'
    else:
        URL_consider = URL + 'page-' + str(page)
    r = requests.get(URL_consider)
    soup = BeautifulSoup(r.content, 'html5lib')
    contents = soup.find('div', class_="block-container california-forum-view-thread-list-container")
    contents = contents.find_all('ul', class_="structItem-parts")
    for article in contents:
        data_thisIter = article.find_all('a', class_ = "start-date")
        links.append('https://www.uberpeople.net' + data_thisIter[0].attrs['href'])
        time_stamp.append(data_thisIter[0].find(class_ = 'u-dt').attrs['datetime'])

toc = time.time()
exe_time = toc - tic
exe_time
len(time_stamp)
len(links)


# %% get the texts from the extracted link articles
articles = []
idx_unuseable = []
len(links)
len(time_stamp)
tic = time.time()
for idx, article in enumerate(links):
    if idx % 50 == 0:
        print(idx)
    r = requests.get(article)
    soup = BeautifulSoup(r.content, 'html5lib')
    contents = soup.find('div', class_= "bbWrapper")
    try:
        contents = contents.find('div', class_="contentRow").decompose()
    except:
        pass
    try:
        articles.append([contents.text])
    except:
        idx_unuseable.append(idx)


assert len(articles) + len(idx_unuseable) == len(links)
toc = time.time()
exe_time = toc - tic
exe_time

len(articles)
len(idx_unuseable)
len(time_stamp)

# %% remove time_stamp associated with empty strings
time_stamp_ = [i for j, i in enumerate(time_stamp) if j not in idx_unuseable]
assert len(time_stamp_) == len(articles)

# %% export the data
os.getcwd()
os.chdir('data')
dict_ ={'articles':articles , 'timestamp': time_stamp_}
with open('data.pickle','wb') as f:
    pickle.dump(dict_,f)
