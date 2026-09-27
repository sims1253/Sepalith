import os,fcntl,json,pathlib,subprocess,sys
fd=int(os.environ['SEPALITH_CUDA_LOCK_FD']);st=os.fstat(fd)
code="import os,fcntl,sys; f=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
r=subprocess.run([sys.executable,'-c',code,sys.argv[1]],capture_output=True,text=True)
assert r.returncode!=0 and 'BlockingIOError' in r.stderr
pathlib.Path(sys.argv[2]).write_text(json.dumps({'inherited_fd':fd,'device':st.st_dev,'inode':st.st_ino,'competing_lock_rejected':True,'CUDA_loaded':False})+'\n')
