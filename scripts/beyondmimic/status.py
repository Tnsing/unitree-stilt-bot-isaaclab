#!/usr/bin/env python3
"""Read actual PID, trainer stdout, checkpoints and evaluation artifacts."""
import json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
state=json.loads((ROOT/'artifacts/beyondmimic/current.json').read_text())
for kind in ['supervisor','training']:
 try:os.kill(state[kind+'_pid'],0);state[kind+'_pid_alive']=True
 except ProcessLookupError:state[kind+'_pid_alive']=False
log=Path(state['log_dir'])/'train.log'
if log.exists():
 text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(errors='replace'))
 iterations=re.findall(r'Learning iteration\s+(\d+)/(\d+)',text)
 if iterations:state['observed_latest_iteration']=int(iterations[-1][0]);state['observed_progress_fraction']=(int(iterations[-1][0])+1)/int(iterations[-1][1])
 for key,pattern in [('observed_total_timesteps',r'Total timesteps:\s+(\d+)'),('observed_eta',r'ETA:\s+(\S+)')]:
  values=re.findall(pattern,text)
  if values:state[key]=values[-1]
print(json.dumps(state,indent=2,ensure_ascii=False))
