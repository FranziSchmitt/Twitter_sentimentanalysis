import json
from collections import Counter

# text processing
with open('stopwords-de_no-parties.json', 'r') as file:
    stopwords_de = json.load(file)
import spacy
nlp = spacy.load('de')
from  nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("german")


def keep_more_tags(text, nlp):
    """
    Keep only verbs, nouns, adjectives and advers that are not in the list of stopwords and longer than two letters
    
    Parameters:
    text: string
    nlp: spacy function loaded in the preamble
    """
    nlp_text = nlp(text)
    wanted = ["ADJA", "ADJD", "NN", "NE", "NNE", 'ADV', 'VERB', "PRON"]
    wanted_tags = [word.text for word in nlp_text if word.tag_ in wanted]
    
    text_out = []
    for word in wanted_tags:
        if ((word.lower() not in stopwords_de)
            and (len(word) > 2)):
            word = stemmer.stem(word)
            text_out.append(word)
    
    return text_out


def sparse_text(text, nlp):
    """
    Keep only verbs, nouns, adjectives and advers that are not in the list of stopwords and longer than two letters
    
    Parameters:
    text: string
    nlp: spacy function loaded in the preamble
    """
    text_out = []
    wanted_tags = keep_tags(text, nlp)
    
    for word in wanted_tags:
        if ((word.lower() not in stopwords_de)
            and (len(word) > 2)):
            word = stemmer.stem(word)
            text_out.append(word)
    
    return text_out
            
def sparse_text_nostemmer(text, nlp):
    """
    Keep only verbs, nouns, adjectives and advers that are not in the list of stopwords and longer than two letters
    
    Parameters:
    text: string
    nlp: spacy function loaded in the preamble
    """
    text_out = []
    wanted_tags = keep_tags(text, nlp)
    
    for word in wanted_tags:
        if ((word.lower() not in stopwords_de)
            and (len(word) > 2)):
            text_out.append(word)
    
    return text_out
       
def keep_tags(text, nlp):
    """
    Only keep nouns (NN), verbs('VERB'), adverbs('ADV') and adjectives(ADJA) from a string
    
    Parameters:
    text: string
    nlp: spacy function loaded in the preamble
    """
    nlp_text = nlp(text)
    wanted = ["ADJA", "NN", 'ADV', 'VERB']
    wanted_words = [word.text for word in nlp_text if word.tag_ in wanted]
    return wanted_words


def party_substituter(x):
    """
    Since the spacy tagger does not recognize party acronyms as nouns, 
    parties are replaced by substitutes to make them available to the model.
    """
    return x.replace('spd', 'sozialdemokraten').replace('cdu', 'christdemokraten').replace('csu', 'christsoziale').replace('fdp', 'freidemokraten').replace('afd', 'pfannen')

def dict_pruning(corpus, cutoff):
    flat_corpus = [item for sublist in corpus for item in sublist]
    counter = Counter(flat_corpus)
    
    few_list = []
    many_list = []
    
    for key, value in counter.items():
        if value < cutoff:
            few_list.append(key)
        else:
            many_list.append(key)
    
    use_many = True if (len(many_list) < len(few_list)) else False
    new_corpus = []
    
    for tweet in corpus:
        new_tweet = []
        for word in tweet:
            if use_many:
                if word in many_list:
                    new_tweet.append(word)
            else:
                if word not in few_list:
                    new_tweet.append(word)
        new_corpus.append(new_tweet.copy())
    
    return new_corpus