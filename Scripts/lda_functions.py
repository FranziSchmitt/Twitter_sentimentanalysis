import json

# text processing
with open('stopwords-de_no-parties.json', 'r') as file:
    stopwords_de = json.load(file)
import spacy
nlp = spacy.load('de')


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
    party_dict = {'AfD': 'Pfannen', 
                  'SPD': 'Teller', 
                  'CDU': 'Messer', 
                  'CSU': 'Gabel', 
                  'FDP': 'Freidemokraten'}
    
    new_text = ''
    text = x.lower()
    
    for acro, new in party_dict.items():
        if acro.lower() in text:
            new_text = text.replace(acro.lower(), new.lower())
        else:
            new_text = text
    return new_text
