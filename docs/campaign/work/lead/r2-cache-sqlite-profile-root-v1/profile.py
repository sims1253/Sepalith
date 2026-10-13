import sqlite3,time,hashlib,json,resource
from pathlib import Path
root=Path('/mnt/e/sepalith/campaign-20260915/tmp/cache-sqlite-profile-root-v1');root.mkdir(exist_ok=False)
rows=[(i,hashlib.sha256(str(i).encode()).hexdigest(), 'x'*400) for i in range(16000)]
results=[]
for setting in (-2000,-65536):
 p=root/f'cache{abs(setting)}.sqlite3';start=time.monotonic();db=sqlite3.connect(p);db.execute('pragma journal_mode=DELETE');db.execute('pragma synchronous=FULL');db.execute(f'pragma cache_size={setting}');db.execute('create table records(i integer primary key,h text unique,payload text)')
 db.executemany('insert into records values(?,?,?)',rows);db.commit();assert db.execute('select count(*) from records').fetchone()[0]==len(rows);db.close()
 results.append({'cache_kib':abs(setting),'seconds':time.monotonic()-start,'rows':len(rows),'bytes':p.stat().st_size})
report={'scope':'synthetic same-host E SQLite random-key insert comparison; not production throughput proof','results':results,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
