import pandas as pd
import numpy as np
from rich import print
import psycopg2

df = pd.read_csv("agri_pub2.csv")

print(df['publication_year'].unique())

print((df['publication_year']==0).value_counts())

#dbname = '{your_database}' user = 'rairo@pubag' host = 'pubag.postgres.database.azure.com' password = '{your_password}' port = '5432' sslmode = 'true'

conn = psycopg2.connect(
    host="pubag.postgres.database.azure.com",
    port="5432",
    database="postgres",
    user="rairo@pubag",
    sslmode="require",
    password="pubag1036!")
cursor = conn.cursor()
