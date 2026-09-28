import json,pathlib,subprocess
w=pathlib.Path(__file__).resolve().parent
for i,cmd in enumerate(json.loads((w/"exports.json").read_text())):
 with (w/(str(i)+".log")).open("xb") as f:
  r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (w/(str(i)+"-terminal.json")).write_text(json.dumps({"exit_code":r.returncode,"command":cmd})+"\n")
 if r.returncode:raise SystemExit(r.returncode)
