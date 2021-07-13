from pymongo import MongoClient
from random import randint
import pandas as pd
import json
import numpy as np
from copy import deepcopy as dp
from scipy.stats import spearmanr
import warnings
from csv import reader

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

keywords = pd.read_csv('hctsa_features.csv')
hctsa = pd.read_csv('hctsa_datamatrix.csv')
hctsa = hctsa.fillna(0)
for col in hctsa.columns:
    hctsa[col] = hctsa[col].astype("float64")
col = hctsa.columns
if hctsa.isnull().sum().sum() > 0:
    print("NaN detected!! Exiting")
    exit()
myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["FeaturesCollection"]
li = list(np.random.permutation(len(col)))[:500]
cnt = 0
for i in li:
    print(cnt, i)
    cnt += 1
    dic = {}
    temp = list(keywords.iloc[i])
    dic["ID"] = int(temp[0])
    dic["NAME"] = temp[1]
    dic["KEYWORDS"] = temp[2]
    temp = list(hctsa[col[i]])
    dic["HCTSA_TIMESERIES_VALUE"] = dp(temp)
    coefList = []
    pval = []
    for j in li:
        coef, p = 0, 0
        try:
            if (hctsa.iloc[:, j].isna().sum()) < 50:
                coef, p = spearmanr(hctsa.iloc[:, j], hctsa.iloc[:, i], nan_policy="omit")
        except:
            print("Exception")
        finally:
            coefList.append(str(format(coef, '.3f')))
            pval.append(str(format(p, '.3f')))
    dic["COEF"] = coefList
    dic['PVALUE'] = pval
    mycol.insert_one(dic)

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

'''
myclient = MongoClient(port=27017)
mydb = myclient["CompEngineFeaturesDatabase"]
mycol = mydb["Temp"]
number = 5718
x = mycol.find_one({'ID': {'$in':[5718]}},{ "_id": 0, "NAME":0, "KEYWORD":0, "ID":0})
print(x)
'''
'''
# Function to add All_Time_Series in MongoDB
def addAllTimeSeries():
    myclient = MongoClient(port=27017)
    mydb = myclient["CompEngineFeaturesDatabase"]
    mycol = mydb["TimeSeries"]
    Alltimeseries = []
    with open('hctsa_timeseries-data.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        li = list(csv_reader)
        for i in li:
            Alltimeseries.append(list(map(float, i)))
    Alltimeseriesname = []
    Alltimeseriescategory = []
    with open('hctsa_timeseries-info.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        li = list(csv_reader)
        for i in li:
            temp = i[2].split(",")
            if ('dynsys' in temp):
                Alltimeseriescategory.append("Ordinary Differential Equation (ODE)")
            elif ('map' in temp):
                Alltimeseriescategory.append("Iterative map (Map)")
            elif ('noise' in temp):
                Alltimeseriescategory.append("Uncorrelated noise (Noise)")
            elif ('synthetic' in temp):
                Alltimeseriescategory.append("Synthetic (Other)")
            else:
                Alltimeseriescategory.append("Real-world")
            Alltimeseriesname.append(i[1])
    for i in range(len(Alltimeseries)):
        dic = {"NAME": Alltimeseriesname[i], "TIMESERIES": Alltimeseries[i], "CATEGORY": Alltimeseriescategory[i]}
        mycol.insert_one(dic)


addAllTimeSeries()


def getAllTimeSeries():
    myclient = MongoClient(port=27017)
    mydb = myclient["CompEngineFeaturesDatabase"]
    mycol = mydb["TimeSeries"]
    Alltimeseries = []
    for x in mycol.find({}, {"_id": 0}):
        Alltimeseries.append(x["TIMESERIES"])


getAllTimeSeries()
'''
