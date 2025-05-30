"""Create database."""
import duckdb

db = duckdb.connect(database="data/data.db")

with open('schema.sql') as f:
    ctq = f.read()

db.sql(ctq)
db.commit()
db.close()
