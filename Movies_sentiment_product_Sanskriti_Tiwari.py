

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[3]:


data=pd.read_csv('IMDB Dataset.csv')
data.head()


# In[4]:


# unique ratings
pd.unique(data['sentiment'])


# In[5]:


data['sentiment'].value_counts()


# In[6]:


stp_words=['is','are','am','has','had','have']
def clean_review(review):
    cleanreview=" ".join(word for word in review.split() if word not in stp_words)
    return cleanreview

data['review']=data['review'].apply(clean_review)
data.head()


# In[7]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['review'] ).toarray()


# In[8]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,data['sentiment'],test_size=0.25,random_state=42)
#check shape 
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[9]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

#Model fitting
model.fit(x_train,y_train)

#testing the model
pred=model.predict(x_test)

#model accuracy
print(accuracy_score(y_test,pred))


# In[10]:


from sklearn import metrics
cm = confusion_matrix(y_test,pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = [False, True])

cm_display.plot()
plt.show()


# In[11]:


import joblib
joblib.dump(stp_words,'stopwords.pkl') 
joblib.dump(model,'model.pkl')
joblib.dump(cv,'vectorizer.pkl')
loaded_model=joblib.load("model.pkl")
loaded_stop=joblib.load("stopwords.pkl")
loaded_vec=joblib.load("vectorizer.pkl")


# In[19]:


# tkinter GUI
import tkinter as tk 
root= tk.Tk()

heading=tk.Label(root,text="MOVIE PREDICTION APP")
heading.pack()

#Make a Canvas (i.e, a screen for your project
canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.configure(bg='pink')
canvas1.pack()

# Outlook label and input box
label1 = tk.Label(root, text=' Enter your movie review : ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(300, 100, window=entry1)

    
import numpy as np
def classify():
 movie_review= entry1.get()
 label = {1: 'Positive', 0: 'Negative'}
 X = loaded_vec.transform([movie_review])
 y = loaded_model.predict(X)
 if [y]==[1]:
    label_Prediction = tk.Label(root, text=y[0], bg='orange')
    canvas1.create_window(200, 230, window=label_Prediction)

 else:
    label_Prediction = tk.Label(root, text=y[0], bg='orange')
    canvas1.create_window(200, 230, window=label_Prediction)



button1=tk.Button(root, text='Predict to watch movie or not',command=classify, bg='orange')
# button to call the 'values' command above
canvas1.create_window(250, 200, window=button1)

#To see the GUI screen
root.mainloop()


# In[ ]:





# In[ ]:




