from typing import Dict
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time
from afinn import Afinn
import pandas as pd

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
afn=Afinn()



for x in range(1000):
   docs = db.collection(u'comments').where(u'bool', u'==', True).stream()
   for doc in docs:  
       doc.to_dict()
       Dict = doc.to_dict()
       c=Dict['Comment']
       ids = str(doc.id)
       
       comments=c
       comments=[c]
       
       scores = [afn.score(article) for article in comments]
       sentiment=['Positive' if score>0
              else 'negative' if score<0
                  else 'neutral' 
                      for score in scores]

       df=pd.DataFrame()
       df['topic'] =comments
       df['scores']=scores
       df['sentiments']=sentiment
       print(ids)
       if sentiment[0]=='negative':
         db.collection(u'comments').document(ids).delete()
       else:
         city_ref = db.collection(u'comments').document(ids)
         city_ref.set({
             u'bool': False
           }, merge=True)
   print("g")
