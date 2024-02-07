import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("DELETE FROM images")
conn.commit()
conn.close()