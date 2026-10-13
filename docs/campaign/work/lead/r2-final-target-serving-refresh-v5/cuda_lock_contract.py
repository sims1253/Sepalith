#!/usr/bin/env python3
"""Validate that an inherited fd is the campaign CUDA lock and its lease is held."""
import fcntl,os,stat
from pathlib import Path

CUDA_LOCK_PATH=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
def require(x,msg):
 if not x:raise ValueError(msg)
def require_lock_fd(fd):
 require(type(fd) is int and fd>=3,'stable CUDA lock fd required')
 owned=os.fstat(fd);expected=CUDA_LOCK_PATH.stat(follow_symlinks=False)
 require(stat.S_ISREG(owned.st_mode) and (owned.st_dev,owned.st_ino)==(expected.st_dev,expected.st_ino),'CUDA lock fd does not identify campaign cuda0.lock')
 probe=os.open(CUDA_LOCK_PATH,os.O_RDWR|os.O_NOFOLLOW)
 try:
  try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:
   fcntl.flock(probe,fcntl.LOCK_UN);raise ValueError('CUDA lock fd identity is correct but exclusive lease is not held')
 finally:os.close(probe)
 return {'path':str(CUDA_LOCK_PATH),'device':owned.st_dev,'inode':owned.st_ino}
