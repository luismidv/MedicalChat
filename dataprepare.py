import pandas as pd
import json
from bs4 import BeautifulSoup
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from torch.utils.data import Dataset,DataLoader

nltk.download('punkt_tab')
X_train = []
Y_train = []
X_test = []
Y_test = []

xy_test = [
    (['ca', "n't", 'think', 'straight'], 'altered_sensorium'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['bloody', 'poop'], 'bloody_stool'),
    (['blurred', 'vision'], 'blurred_and_distorted_vision'),
    (['ca', "n't", 'breathe'], 'breathlessness'),
    (['Yellow', 'liquid', 'pimple'], 'yellow_crust_ooze'),
    (['lost', 'weight'], 'weight_loss'),
    (['side', 'weaker'], 'weakness_of_one_body_side'),
    (['watering', 'eyes'], 'watering_from_eyes'),
    (['brief', 'blindness'], 'visual_disturbances'),
    (['throat', 'hurts'], 'throat_irritation'),
    (['extremities', 'swelling'], 'swollen_extremeties'),
    (['swollen', 'lymph', 'nodes'], 'swelled_lymph_nodes'),
    (['dark', 'under', 'eyes'], 'sunken_eyes'),
    (['stomach', 'blood'], 'stomach_bleeding'),
    (['blood', 'urine'], 'spotting_urination'),
    (['sinuses', 'hurt'], 'sinus_pressure'),
    (['watery', 'from', 'nose'], 'runny_nose'),
    (['have', 'to', 'move'], 'restlessness'),
    (['red', 'patches', 'body'], 'red_spots_over_body'),
    (['sneeze'], 'continuous_sneezing'),
    (['coughing'], 'cough'),
    (['skin', 'patches'], 'dischromic_patches'),
    (['skin', 'bruised'], 'bruising'),
    (['burning', 'pee'], 'burning_micturition'),
    (['hurts', 'pee'], 'burning_micturition'),
    (['Burning', 'sensation'], 'burning_micturition'),
    (['chest', 'pressure'], 'chest_pain'),
    (['pain', 'butt'], 'pain_in_anal_region'),
    (['heart', 'bad', 'beat'], 'palpitations'),
    (['fart', 'lot'], 'passage_of_gases'),
    (['cough', 'phlegm'], 'phlegm'),
    (['lot', 'urine'], 'polyuria'),
    (['Veins', 'bigger'], 'prominent_veins_on_calf'),
    (['Veins', 'emphasized'], 'prominent_veins_on_calf'),
    (['yellow', 'pimples'], 'pus_filled_pimples'),
    (['red', 'nose'], 'red_sore_around_nose'),
    (['skin', 'yellow'], 'yellowish_skin'),
    (['eyes', 'yellow'], 'yellowing_of_eyes'),
    (['large', 'thyroid'], 'enlarged_thyroid'),
    (['really', 'hunger'], 'excessive_hunger'),
    (['always', 'hungry'], 'excessive_hunger'),
]
stemmer = PorterStemmer()
def read_data():
    #TODO FIRST OPTION
    file = 'data/intents.json'
    data = pd.read_json(file)
    
    #SECOND OPTION
    with open(file) as f:
        data1 = json.load(f)
    

    process_data(data)

def process_data(data):
    #SAVE IN PATTERNS EVERY PATTERN WE HAVE FOR A SPECIFIC TASK TOKENIZED
    #FEATURE_LABEL WILL STORE EACH PATTERS, AND THE TAG THEY BELONG TO
    #IN TAGS WE WILL STORE EACH PROBLEM THE PATTERN BELONG TO
    tags = []
    patterns = []
    feature_label = []

    for intents in data['intents']:
        tags.append(intents['tag'])
        for pattern in intents['patterns']:
            w = nltk.word_tokenize(pattern)
            patterns.extend(w)
            feature_label.append((w,intents['tag']))
    new_function(patterns,feature_label,tags)

def new_function(patterns,feature_label,tags):
    #HERE WE WILL STEM THE WORDS TO REDUCE COMPUTATION AND DELETE MARKS, POINTS AND COMMAS

    words_ignore = ['?', '!', '.', ',']
    patterns = [stemmer.stem(word.lower()) for word in patterns if word not in words_ignore]
    patterns = sorted(set(patterns))
    tags = sorted(tags)
    prepare_training_data(patterns,feature_label,tags,X_train, Y_train)    
    prepare_training_data(patterns,xy_test,tags, X_test, Y_test)    


def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence = nltk.word_tokenize(tokenized_sentence)
    tokenized_sentence = [stemmer.stem(word.lower()) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            
            bag[idx] = 1.0
    return bag

def prepare_training_data(patterns_words,feature_label,tags,X,Y):
    
    for (pattern,tag) in feature_label:
        
        bag = bag_of_words(str(pattern),patterns_words)
        X.append(bag)
        label = tags.index(tag)
        Y.append(label)
    X = np.array(X)
    Y = np.array(Y)

read_data()

class ChatBotDataset(Dataset):
    def __init__(self,feature,label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, index):
        feature = feature[index]
        label = label[index]
        return feature,label

train_data = ChatBotDataset(X_train,Y_train)
test_data = ChatBotDataset(X_test,Y_test)
trainChatLoader = DataLoader(train_data, batch_size = 64, shuffle=True)
testChatLoader = DataLoader(test_data, batch_size = 64, shuffle=True)




