"""One exact RTX ledger row; caller holds the stable shared CUDA lock."""
from pathlib import Path
import datetime as dt, hashlib, os, stat, tempfile, uuid
PREFIX=b'| RTX 5090 |'
LIMIT=4*1024*1024

def snapshot(path):
    path=Path(path)
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid():raise RuntimeError('CUDA_ledger_not_owned_regular_file')
    with path.open('rb') as f:data=f.read(LIMIT+1)
    if len(data)>LIMIT:raise RuntimeError('CUDA_ledger_too_large')
    rows=data.splitlines(keepends=True);matches=[i for i,row in enumerate(rows) if row.startswith(PREFIX)]
    if len(matches)!=1:raise RuntimeError('CUDA_ledger_row_missing_or_ambiguous')
    return before,data,rows,matches[0]

def replace_exact(path,expected,replacement):
    path=Path(path);before,data,rows,index=snapshot(path)
    if rows[index]!=expected:raise RuntimeError('CUDA_lease_row_changed')
    rows[index]=replacement;result=b''.join(rows)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            os.fchmod(f.fileno(),stat.S_IMODE(before.st_mode));f.write(result);f.flush();os.fsync(f.fileno())
        current,now,_,_=snapshot(path)
        if (current.st_dev,current.st_ino)!=(before.st_dev,before.st_ino) or now!=data:raise RuntimeError('CUDA_ledger_changed_before_replace')
        os.replace(temp,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if Path(temp).exists():Path(temp).unlink()
    return hashlib.sha256(result).hexdigest()

class LedgerLease:
    def __init__(self,path,instance,supervisor):
        if str(uuid.UUID(instance))!=instance:raise ValueError('invalid_session_UUID')
        self.path=Path(path);self.instance=instance;self.supervisor=supervisor;self.owned_row=None
    def acquire(self):
        if self.owned_row is not None:raise RuntimeError('CUDA_lease_already_acquired')
        _,_,rows,index=snapshot(self.path);free=rows[index]
        if free.split(b'|')[2].strip()!=b'None':raise RuntimeError('CUDA_lease_busy_or_unknown')
        newline=b'\r\n' if free.endswith(b'\r\n') else b'\n' if free.endswith(b'\n') else b''
        now=dt.datetime.now(dt.timezone.utc).isoformat()
        owned=(f'| RTX 5090 | daily-lan | RUN-01 daily-{self.instance}; supervisor {self.supervisor["pid"]}/{self.supervisor["startTick"]} | {now} | This exact session releases after owned children exit |').encode()+newline
        digest=replace_exact(self.path,free,owned);self.owned_row=owned
        return {'status':'acquired','instanceId':self.instance,'supervisor':self.supervisor,'ownedRow':owned.decode().rstrip('\r\n'),'ledgerSha256':digest}
    def monitor(self):
        if self.owned_row is None:raise RuntimeError('CUDA_lease_not_acquired')
        _,_,rows,index=snapshot(self.path)
        if rows[index]!=self.owned_row:raise RuntimeError('CUDA_lease_row_changed')
    def release(self,children_released):
        if self.owned_row is None:return {'status':'not_acquired','ledgerChanged':False}
        if not children_released:return {'status':'retained_owned_children_unreleased','ledgerChanged':False}
        try:
            self.monitor()
            newline=b'\r\n' if self.owned_row.endswith(b'\r\n') else b'\n' if self.owned_row.endswith(b'\n') else b''
            now=dt.datetime.now(dt.timezone.utc).isoformat()
            free=(f'| RTX 5090 | None | RUN-01 daily-{self.instance} terminal; owned children released | {now} | Root admission or admitted daily session before next CUDA job |').encode()+newline
            digest=replace_exact(self.path,self.owned_row,free)
            return {'status':'released','ledgerChanged':True,'ledgerSha256':digest}
        except (OSError,RuntimeError) as exc:
            return {'status':'retained_ledger_mismatch','ledgerChanged':False,'reason':str(exc)}
