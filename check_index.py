import sqlite3
conn = sqlite3.connect(r'E:\project\pybroker\stock_kline_cache.db')
cursor = conn.execute("SELECT date, open, high, low, close, volume, pctChg FROM daily_kline WHERE code = 'sh.000905' ORDER BY date LIMIT 5")
print('First 5 rows:')
for row in cursor.fetchall():
    print(row)
conn.close()
