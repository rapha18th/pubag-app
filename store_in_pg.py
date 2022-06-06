import psycopg2
import json
import pandas as pd
import psycopg2.extras
import sqlalchemy
from sqlalchemy import create_engine
from rich import print


#dbname = '{your_database}' user = 'rairo@pubag' host = 'pubag.postgres.database.azure.com' password = '{your_password}' port = '5432' sslmode = 'true'

conn = psycopg2.connect(
    host="pubag.postgres.database.azure.com",
    port="5432",
    database="postgres",
    user="rairo@pubag",
    sslmode="require",
    password="pubag1036!")
cursor = conn.cursor()
# Print PostgreSQL Connection properties
print(conn.get_dsn_parameters(), "\n")

df1 = pd.read_csv("agri_pub2.csv")

df = df1[['title', 'id', 'author', 'url', 'publication_year', 'abstract', 'processed_abstract', 'label_k20']]

print(list(df))
'''
engine = create_engine(
    'postgresql://rairo@pubag:pubag1036!@pubag.postgres.database.azure.com:5432/postgres')
df.to_sql('pubag_table', engine, if_exists='replace',
          dtype={
                 'title': sqlalchemy.types.VARCHAR(10000),
                 'id': sqlalchemy.types.VARCHAR(10000),
                 'abstract': sqlalchemy.types.VARCHAR(40000),
                 'url': sqlalchemy.types.VARCHAR(40000),
                 'processed_abstract': sqlalchemy.types.VARCHAR(40000),
                 'author': sqlalchemy.types.VARCHAR(10000),
                 'publication_year': sqlalchemy.types.INTEGER(),
                 'label_k20': sqlalchemy.types.INTEGER()
                 })
'''
test_Query = "SELECT * FROM pubag_table LIMIT 10"
cursor.execute(test_Query)
test_result = cursor.fetchone()
print("\n hello Postgres:", test_result)

test_Query2 = "SELECT * FROM search_queries"
cursor.execute(test_Query2)
test_result2 = cursor.fetchone()
print("\n hello Postgres:", test_result2)
