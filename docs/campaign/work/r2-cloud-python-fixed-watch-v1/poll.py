import subprocess,json,datetime,pathlib
w=pathlib.Path(__file__).parent
t=datetime.datetime.now(datetime.timezone.utc).isoformat()
id="prodjob_p5l8wmtwavqwk1v1c6waphm7zm"
out={"at":t,"job_id":id}
for key,args in [("status",["status","--id",id,"-o","json"]),("logs",["logs","--id",id,"--tail","--max-lines","60"])]:
 try:
  p=subprocess.run(["/home/m0hawk/.local/bin/anyscale","job"]+args,capture_output=True,text=True,timeout=25)
  if key=="status":
   d=json.loads(p.stdout);out[key]={k:d.get(k) for k in ["id","name","state","runs","created_at","updated_at"]}
  else:
   events=[]
   for line in p.stdout.splitlines():
    if line.startswith("{"):
     try: events.append(json.loads(line))
     except ValueError: pass
   out[key]={"returncode":p.returncode,"events":events}
 except Exception as e: out[key]={"error_type":type(e).__name__}
(w/("poll-"+t.replace(":","-")+".json")).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"at":t,"status":out.get("status"),"latest_events":out.get("logs",{}).get("events",[])[-1:]},indent=2))
