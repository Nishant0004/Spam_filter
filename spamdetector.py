import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import svm
import pandas as pd
import pickle
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("spam.csv")
message_X = df.iloc[:,1] 
labels_Y = df.iloc[:,0]  

#stemming variable initialization
lstem = LancasterStemmer()
def mess(messages):
  message_x = []
  for me_x in messages:
    me_x = str(me_x)
    me_x=''.join(filter(lambda mes:(mes.isalpha() or mes==" ") ,me_x)) 
    words = word_tokenize(me_x)
    message_x+=[' '.join([lstem.stem(word) for word in words])]
  return message_x

message_x = mess(message_X)
#vectorization process
tfvec=TfidfVectorizer(stop_words='english')
x_new=tfvec.fit_transform(message_x).toarray()

#replace ham and spam with 0 and 1 respectively
y_new = np.array(labels_Y.replace(to_replace=['ham', 'spam'], value=[0, 1]))

x_train , x_test , y_train , y_test = ttsplit(x_new,y_new,test_size=0.2,shuffle=True) 
classifier = svm.SVC()
classifier.fit(x_train,y_train)

pickle.dump({'classifier':classifier,'message_x':message_x},open("training_data.pkl","wb"))
