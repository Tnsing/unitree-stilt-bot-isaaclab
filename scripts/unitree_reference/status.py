#!/usr/bin/env python3
"""Read actual queue state, training logs, checkpoints, audits and video files."""
import json,os,re
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
progress=ROOT/'artifacts/unitree_reference/progress.json'
state=json.loads(progress.read_text()) if progress.exists() else {'status':'not_started','skills':{}}
def alive(pid):
    try:os.kill(pid,0);return True
    except (OSError,TypeError):return False
result={'checked_at_local':datetime.now().astimezone().isoformat(),'suite_status':state['status'],'suite_pid':state.get('pid'),'suite_alive':alive(state.get('pid')),'active_skill':state.get('active_skill'),'skills':{}}
for skill in ['walk','run','dance','stilt']:
    s=dict(state.get('skills',{}).get(skill,{}));log=ROOT/f'logs/unitree_reference/{skill}.train.log'
    if log.exists():
        text=log.read_text(errors='replace');text=re.sub(r'\x1b\[[0-9;]*m','',text);matches=re.findall(r'Learning iteration\s+(\d+)/(\d+)',text)
        if matches:s.update(last_logged_iteration=int(matches[-1][0]),iteration_budget=int(matches[-1][1]))
    if s.get('run'):
        checkpoints=list(Path(s['run']).glob('model_*.pt'));s['checkpoint_count']=len(checkpoints)
        if checkpoints:s['last_checkpoint']=str(max(checkpoints,key=lambda p:int(p.stem.split('_')[-1])))
    s['videos']=[str(p) for p in sorted((ROOT/f'artifacts/unitree_reference/{skill}').glob('video_*/demo.mp4')) if p.stat().st_size>1024]
    result['skills'][skill]=s
print(json.dumps(result,indent=2,ensure_ascii=False))
