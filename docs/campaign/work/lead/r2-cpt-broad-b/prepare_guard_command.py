import sys,json,pathlib
p=pathlib.Path(sys.argv[5]);p.write_text(json.dumps([sys.argv[1],sys.argv[2],sys.argv[3],"--receipt",sys.argv[4]])+"\n")
