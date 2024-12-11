import pandas as pd
import json
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt_tab')

def read_data():
    #TODO FIRST OPTION
    file = 'data/intents.json'
    data = pd.read_json(file)
    
    #SECOND OPTION
    with open(file) as f:
        data1 = json.load(f)
    

    process_data(data)

def process_data(data):
    tags = []
    patterns = []
    feature_label = []

    for intents in data['intents']:
        tags.append(intents['tag'])
        for pattern in intents['patterns']:
            w = nltk.word_tokenize(pattern)
            patterns.extend(w)
            feature_label.append((w,intents['tag']))
    #print(f"Tags {tags} ")
    print(f"patterns {patterns}")
    #print(f"feature_label {feature_label}")


read_data()



