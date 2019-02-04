import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model

model=load_model('LSTM500.h5')
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)

while(True):
    x=str(input("Enter the message:"));
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [model[w] for w in sent if w in model.wv.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<20:
        for i in range(20-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(20)]
    output=' '.join(outputlist)
    print(output)