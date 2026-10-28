#!/usr/bin/env python3
"""Serial formal training + milestone policy video + independent final audit.

Run via runtime.sh Python. Locking prevents duplicate GPU training suites.
"""
import argparse,fcntl,json,os,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--adopt-walk',type=Path);p.add_argument('--adopt-pid',type=int);p.add_argument('--iterations',type=int,default=2000);p.add_argument('--num-envs',type=int,default=4096);args=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/unitree_reference';BASE.mkdir(parents=True,exist_ok=True)
lock=(BASE/'suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'started_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'runtime':'Lab 2.3 / Sim 5.1 / PhysX, explicit compatibility port of pinned Lab 3.0 source','budget':args.iterations,'num_envs':args.num_envs,'skills':{},'status':'running'}
def save():
 state['updated_at']=datetime.now(timezone.utc).isoformat();f=BASE/'progress.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(f)
def command(cmd,log,env=None):
 print('RUN',cmd,'LOG',log,flush=True)
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);state['child_pid']=proc.pid;save()
  rc=proc.wait()
 if rc:raise RuntimeError(f'Command exit {rc}: {log}')
state['skills']={s:{'status':'queued'} for s in ['walk','run','dance','stilt']};save()
try:
 validation=json.loads((BASE/'stilt_physics/validation.json').read_text());assert validation['passed']
 for skill in ['walk','run','dance','stilt']:
  s=state['skills'][skill];state['active_skill']=skill;s['status']='training';save()
  if skill=='walk' and args.adopt_walk:
   run=args.adopt_walk.resolve();s['run']=str(run);save()
   if args.adopt_pid:
    while True:
     try:os.kill(args.adopt_pid,0)
     except ProcessLookupError:break
     time.sleep(5)
   assert (run/f'model_{args.iterations-1}.pt').exists(),f'Adopted training did not finish: {run}'
  else:
   env=os.environ.copy();env.update(SKILL=skill,NUM_ENVS=str(args.num_envs),MAX_ITERATIONS=str(args.iterations),RUN_NAME='formal_unitree_ref_seed42',SEED='42')
   command([str(ROOT/'scripts/unitree_reference/train.sh')],ROOT/f'logs/unitree_reference/{skill}.train.log',env)
   dirs=[d for d in (ROOT/f'logs/rsl_rl/unitree_reference_{skill}').glob('*formal_unitree_ref_seed42') if d.is_dir()]
   run=max(dirs,key=lambda d:d.stat().st_mtime);s['run']=str(run)
  s['status']='milestone_videos';save()
  for iteration in [500,1000,args.iterations-1]:
   cp=run/f'model_{iteration}.pt'
   if not cp.exists():continue
   output=BASE/skill/f'video_{iteration}'
   if (output/'evaluation.json').exists():continue
   command(['python',str(ROOT/'scripts/unitree_reference/rollout.py'),'--skill',skill,'--num_envs','1','--steps','500','--checkpoint',str(cp),'--output',str(output),'--video','--headless'],ROOT/f'logs/unitree_reference/{skill}.video_{iteration}.log')
  s['status']='independent_audit';save()
  output=BASE/skill/'audit_final'
  if not (output/'evaluation.json').exists():
   command(['python',str(ROOT/'scripts/unitree_reference/rollout.py'),'--skill',skill,'--num_envs','64','--steps','1500','--checkpoint',str(run/f'model_{args.iterations-1}.pt'),'--output',str(output),'--headless'],ROOT/f'logs/unitree_reference/{skill}.audit_final.log')
  command(['python',str(ROOT/'scripts/unitree_reference/collect_milestone.py'),'--skill',skill,'--run',str(run),'--budget',str(args.iterations),'--out',str(BASE/skill/'milestone.json')],ROOT/f'logs/unitree_reference/{skill}.manifest.log')
  audit=json.loads((output/'evaluation.json').read_text());s['status']='trained_and_audited';s['gate']=audit['gate'];save()
 state['status']='completed_budget_and_audits';state['all_gates_passed']=all(s.get('gate',{}).get('passed',False) for s in state['skills'].values());save()
except BaseException as exc:
 state['status']='failed';state['error']=repr(exc);save();traceback.print_exc();raise
