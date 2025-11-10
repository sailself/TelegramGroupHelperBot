import sqlite3, json

db_path = "bot.db"
con = sqlite3.connect(db_path)
cur = con.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
result = {}
for t in tables:
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
    if 'chat_id' in cols:
        rows = list(cur.execute(f"SELECT chat_id, COUNT(*) FROM {t} GROUP BY chat_id ORDER BY COUNT(*) DESC"))
        result[t] = rows
print(json.dumps(result, ensure_ascii=False))
