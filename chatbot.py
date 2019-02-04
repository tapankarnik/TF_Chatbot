# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
import gensim
import nltk 
import pickle
 
########## PART 1 - DATA PREPROCESSING ##########
 
 
 
# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
 
# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
 
# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

questions = questions[:100]
answers = answers[:100]
 
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text
 
# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
 
# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
 
# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1

tok_question = []
tok_answers = []
for i in range(len(short_questions)):
    tok_question.append(nltk.word_tokenize(short_questions[i].lower()))
    tok_answers.append(nltk.word_tokenize(short_answers[i].lower()))

vec_question=[]
for sent in tok_question:
    sentvec = [model[w] for w in sent if w in model.wv.vocab]
    vec_question.append(sentvec)

vec_answer=[]
for sent in tok_answers:
    sentvec = [model[w] for w in sent if w in model.wv.vocab]
    vec_answer.append(sentvec)

setend = np.ones((300,),dtype=np.float32)
for tok_sent in vec_question:
    tok_sent[19:]=[]
    tok_sent.append(setend)

for tok_sent in vec_question:
    if(len(tok_sent)<20):
        for i in range(20-len(tok_sent)):
            tok_sent.append(setend)

for tok_sent in vec_answer:
    tok_sent[19:]=[]
    tok_sent.append(setend)

for tok_sent in vec_answer:
    if(len(tok_sent)<20):
        for i in range(20-len(tok_sent)):
            tok_sent.append(setend)

with open("conversation.pickle","wb") as f:
    pickle.dump([vec_question,vec_answer],f)
    

##################################################################
    
    
    

