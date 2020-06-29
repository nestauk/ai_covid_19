#Various functions and utilities that we use to work with text
import re
import string
from string import punctuation
from string import digits
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import *

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
stemmer = PorterStemmer()

stop_words = set(
    stopwords.words("english") + list(string.punctuation) + ["\\n"] + ["quot"]
)

regex_str = [
    r"http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|" r"[!*\(\),](?:%[0-9a-f][0-9a-f]))+",
    r"(?:\w+-\w+){2}",
    r"(?:\w+-\w+)",
    r"(?:\\\+n+)",
    r"(?:@[\w_]+)",
    r"<[^>]+>",
    r"(?:\w+'\w)",
    r"(?:[\w_]+)",
    r"(?:\S)",
]

# Create the tokenizer which will be case insensitive and will ignore space.
tokens_re = re.compile(r"(" + "|".join(regex_str) + ")", re.VERBOSE | re.IGNORECASE)

stemmer = PorterStemmer()

def tokenize_document(text, remove_stops=False):
    """Preprocess a whole raw document.

    Args:
        text (str): Raw string of text.
        remove_stops (bool): Flag to remove english stopwords

    Return:
        List of preprocessed and tokenized documents

    """
    return [
        clean_and_tokenize(sentence, remove_stops)
        for sentence in nltk.sent_tokenize(text)
    ]

def clean_and_tokenize(text, remove_stops):
    """Preprocess a raw string/sentence of text.

    Args:
       text (str): Raw string of text.
       remove_stops (bool): Flag to remove english stopwords

    Return:
       tokens (list, str): Preprocessed tokens.

    """
    tokens = tokens_re.findall(text)
    _tokens = [t.lower() for t in tokens]
    filtered_tokens = [
        token.replace("-", "_")
        for token in _tokens
        if not (remove_stops and len(token) <= 2)
        and (not remove_stops or token not in stop_words)
        and not any(x in token for x in string.digits)
        and any(x in token for x in string.ascii_lowercase)
    ]
    return filtered_tokens


def tfidf_vectors(data, max_features):
    """Transforms text to tfidf vectors.

    Args:
        data (pandas.Series)
    
    Returns:
        (`scipy.sparse`): Sparse TFIDF matrix.

    """
    vectorizer = TfidfVectorizer(
        stop_words="english", analyzer="word", max_features=max_features
    )
    return vectorizer.fit_transform(data)

#Characters to drop
drop_characters = re.sub('-','',punctuation)+digits

def clean_tokenise(string,drop_characters=drop_characters,stopwords=stop_words):
    '''
    Takes a string and cleans (makes lowercase and removes stopwords)
    
    '''
    #Lowercase
    str_low = string.lower()
    
    
    #Remove symbols and numbers
    str_letters = re.sub('[{drop}]'.format(drop=drop_characters),'',str_low)
    
    
    #Remove stopwords
    clean = [x for x in str_letters.split(' ') if (x not in stopwords) & (x!='')]
    
    return(clean)


class CleanTokenize():
    '''
    This class takes a list of strings and returns a tokenised, clean list of token lists ready
    to be processed with the LdaPipeline
    
    It has a clean method to remove symbols and stopwords
    
    It has a bigram method to detect collocated words
    
    It has a stem method to stem words
    
    '''
    
    def __init__(self,corpus):
        '''
        Takes a corpus (list where each element is a string)
        '''
        
        #Store
        self.corpus = corpus
        
    def clean(self,drop=drop_characters,stopwords=stop_words):
        '''
        Removes strings and stopwords, 
        
        '''     
        cleaned = [clean_tokenise(doc,drop_characters=drop,stopwords=stopwords) for doc in self.corpus]
        
        self.tokenised = cleaned
        return(self)
    
    def stem(self):
        '''
        Optional: stems words
        
        '''
        #Stems each word in each tokenised sentence
        stemmed = [[stemmer.stem(word) for word in sentence] for sentence in self.tokenised]
    
        self.tokenised = stemmed
        return(self)
        
    
    def bigram(self,threshold=10):
        '''
        Optional Create bigrams.
        
        '''
        
        #Colocation detector trained on the data
        phrases = models.Phrases(self.tokenised,threshold=threshold)
        
        bigram = models.phrases.Phraser(phrases)
        
        self.tokenised = bigram[self.tokenised]
        
        return(self)
        
def salient_words_per_category(token_df,corpus_freqs,thres=100,top_words=50):
    '''
    Create a list of salient terms in a df (salient terms normalised by corpus frequency).
    
    Args:
        tokens (list or series) a list where every element is a tokenised abstract
        corpus_freqs (df) are the frequencies of terms in the whole corpus
        thres (int) is the number of occurrences of a term in the subcorpus
        top_words (int) is the number of salient words to output
    
    '''
    
    subcorpus_freqs = flatten_freq(token_df,freq=True)
    
    merged= pd.concat([pd.DataFrame(subcorpus_freqs),corpus_freqs],axis=1,sort=True)
    
    merged['salience'] = (merged.iloc[:,0]/merged.iloc[:,1])
    
    
    results = merged.loc[merged.iloc[:,0]>thres].sort_values('salience',ascending=False).iloc[:top_words]
    
    results.columns = ['sub_corpus','corpus','salience']
    
    return results


def get_term_salience(df,sel_var,sel_term,corpus_freqs,thres=100,top_words=50):
    '''
    Returns a list of salient terms per SDG
    
    Args:
        df (df) is a df of interest
        sel_var (str) is the variable we use to select
        sel_term (str) is the term we use to select
        corpus_freqs (df) is a df with corpus frequencies
        thres (int) is the min number of word occurrences
        top_words (int) is the number of words to report

    '''
    
    rel_corp = df.loc[df[sel_var]==sel_term].drop_duplicates('project_id')['tokenised_abstract']
    
    salient_rel = salient_words_per_category(list(rel_corp),corpus_freqs,thres,top_words)
    
    salient_rel.rename(columns={'sub_corpus':f'{str(sel_term)}_freq','corpus':'all_freq',
                               'salience':f'{str(sel_term)}_salience'},inplace=True)
    
    return(salient_rel)
    
class LdaPipeline():
    '''
    This class processes lists of keywords.
    How does it work?
    -It is initialised with a list where every element is a collection of keywords
    -It has a method to filter keywords removing those that appear less than a set number of times
    
    -It has a method to process the filtered df into an object that gensim can work with
    -It has a method to train the LDA model with the right parameters
    -It has a method to predict the topics in a corpus
    
    '''
    
    def __init__(self,corpus):
        '''
        Takes the list of terms
        '''
        
        #Store the corpus
        self.tokenised = corpus
        
    def filter(self,minimum=5):
        '''
        Removes keywords that appear less than 5 times.
        
        '''
        
        #Load
        tokenised = self.tokenised
        
        #Count tokens
        token_counts = pd.Series([x for el in tokenised for x in el]).value_counts()
        
        #Tokens to keep
        keep = token_counts.index[token_counts>minimum]
        
        #Filter
        tokenised_filtered = [[x for x in el if x in keep] for el in tokenised]
        
        #Store
        self.tokenised = tokenised_filtered
        self.empty_groups = np.sum([len(x)==0 for x in tokenised_filtered])
        
        return(self)
    
    def clean(self):
        '''
        Remove symbols and numbers
        
        '''
         
    def process(self):
        '''
        This creates the bag of words we use in the gensim analysis
        
        '''
        #Load the list of keywords
        tokenised = self.tokenised
        
        #Create the dictionary
        dictionary = corpora.Dictionary(tokenised)
        
        #Create the Bag of words. This converts keywords into ids
        corpus = [dictionary.doc2bow(x) for x in tokenised]
        
        self.corpus = corpus
        self.dictionary = dictionary
        return(self)
        
    def tfidf(self):
        '''
        This is optional: We extract the term-frequency inverse document frequency of the words in
        the corpus. The idea is to identify those keywords that are more salient in a document by normalising over
        their frequency in the whole corpus
        
        '''
        #Load the corpus
        corpus = self.corpus
        
        #Fit a TFIDF model on the data
        tfidf = models.TfidfModel(corpus)
        
        #Transform the corpus and save it
        self.corpus = tfidf[corpus]
        
        return(self)
    
    def fit_lda(self,num_topics=20,passes=5,iterations=75,random_state=1803):
        '''
        
        This fits the LDA model taking a set of keyword arguments.
        #Number of passes, iterations and random state for reproducibility. We will have to consider
        reproducibility eventually.
        
        '''
        
        #Load the corpus
        corpus = self.corpus
        
        #Train the LDA model with the parameters we supplied
        lda = models.LdaModel(corpus,id2word=self.dictionary,
                              num_topics=num_topics,passes=passes,iterations=iterations,random_state=random_state)
        
        #Save the outputs
        self.lda_model = lda
        self.lda_topics = lda.show_topics(num_topics=num_topics)
        

        return(self)
    
    def predict_topics(self):
        '''
        This predicts the topic mix for every observation in the corpus
        
        '''
        #Load the attributes we will be working with
        lda = self.lda_model
        corpus = self.corpus
        
        #Now we create a df
        predicted = lda[corpus]
        
        #Convert this into a dataframe
        predicted_df = pd.concat([pd.DataFrame({x[0]:x[1] for x in topics},
                                              index=[num]) for num,topics in enumerate(predicted)]).fillna(0)
        
        self.predicted_df = predicted_df
        
        return(self)
