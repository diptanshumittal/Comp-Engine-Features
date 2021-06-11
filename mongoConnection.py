from pymongo import MongoClient
from random import randint
import pandas as pd
import json
import numpy as np
from copy import deepcopy as dp
from scipy.stats import spearmanr
import warnings
   
'''
warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)


keywords = pd.read_csv('hctsa_features.csv')
hctsa = pd.read_csv('hctsa_datamatrix.csv')
myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["Temp2"]
li = list(np.random.permutation(7702))[:500]
cnt = 0
for i in li:
  print(cnt)
  cnt+=1
  dic = {}
  temp = list(keywords.iloc[i])
  dic["ID"] = int(temp[0])
  dic["NAME"] = temp[1]
  dic["KEYWORDS"] = temp[2]
  temp = list(hctsa.iloc[:,i])
  dic["HCTSA_TIMESERIES_VALUE"] = dp(temp)
  coefList = []
  pval = []
  for j in li:
    coef , p =0 ,0
    if (hctsa.iloc[:,j].isna().sum())<50:
      coef, p = spearmanr(hctsa.iloc[:,j],temp,nan_policy="omit")
    coefList.append(str(format(coef,'.3f')))
    pval.append(str(format(p,'.3f')))
  dic["COEF"] = coefList
  dic['PVALUE'] = pval
  mycol.insert_one(dic)
'''

'''
myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["Temp"]
for x in mycol.find({},{ "_id": 0}):
  #li = list(map(np.int64, x["PVALUE"]))
  print(x["PVALUE"][10])
  li = np.fromstring(x["PVALUE"][10], dtype=np.float64, sep=' ')
  print(li)
  break
'''
"""
keywords = pd.read_csv('hctsa_features.csv')
#hctsa = pd.read_csv('hctsa_datamatrix.csv')

result = keywords.to_json(orient="index")
result = json.loads(result)
li = []
for i in list(result.keys()):
    li.append(result[i])

myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["FeaturesCollection"]
x = mycol.insert_many(li)


myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["FeaturesCollection"]
for x in mycol.find({},{ "_id": 0}):
  print(x)
"""

myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["Temp"]
number = 5718
x = mycol.find_one({'ID': {'$in':[5718]}},{ "_id": 0, "NAME":0, "KEYWORD":0, "ID":0})
print(x)