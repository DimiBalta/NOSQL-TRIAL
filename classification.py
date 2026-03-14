import pandas as pd
import numpy as np
from pymongo import MongoClient

# connect to MongoDBAtlas
client = MongoClient("mongodb://dimibalta:Ioanna05!@ac-ccodocm-shard-00-00.lelleyd.mongodb.net:27017,ac-ccodocm-shard-00-01.lelleyd.mongodb.net:27017,ac-ccodocm-shard-00-02.lelleyd.mongodb.net:27017/?ssl=true&replicaSet=atlas-v0cv35-shard-0&authSource=admin&appName=Cluster0")

#select the database
db = client ['analytics'] #database
collection = db["clickstream"] #inside the database

#retrieve documents 
data = list(collection.find())
df = pd.DataFrame(data)

print(df.head())

#overview of descriptive statistics
print(df.describe())

#check data types
print(df.info())