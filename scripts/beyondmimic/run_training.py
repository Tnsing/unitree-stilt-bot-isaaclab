#!/usr/bin/env python3
"""Durable official-MDP training supervisor with serialized milestone evaluation.

Training runs in one uninterrupted RSL-RL process; an independent worker loads
immutable checkpoints on the same GPU. A single evaluation at a time bounds VRAM.
"""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=30000);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
BASE=ROOT/'artifacts/beyondmimic';BASE.mkdir(exist_ok=True)
lock=(BASE/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
name='formal_official_mdp_seed'+str(args.seed)+'_'+stamp
session=BASE/name;session.mkdir()
logs=ROOT/'logs/beyondmimic'/name;logs.mkdir(parents=True)
state={'status':'starting','supervisor_pid':os.getpid(),'created_at':datetime.now(timezone.utc).isoformat(),'task':'BeyondMimic-G1-29dof-Dance-v0','iterations':args.iterations,'num_envs':args.num_envs,'seed':args.seed,'session_dir':str(session),'log_dir':str(logs),'checkpoints_every_iterations':500,'milestones':{},'run_name':name,'training_completed':False,'task_gate_passed':None}
def save():
 state['updated_at']=datetime.now(timezone.utc).isoformat()
 for dst in [session/'progress.json',BASE/'current.json']:
  tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(dst)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
provenance={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_status_at_launch':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),'source_sha256':{str(f.relative_to(ROOT)):sha(f) for folder in ['src','scripts/beyondmimic','vendor/whole_body_tracking'] for f in (ROOT/folder).rglob('*.py')},'lock':json.loads((ROOT/'configs/beyondmimic_lock.json').read_text()),'motion':json.loads((ROOT/'assets/motions/beyondmimic_dance_102_50hz.json').read_text()),'software_smoke_only':'32 env x 10 iterations; not a learned policy result'}
(session/'launch.json').write_text(json.dumps(provenance,indent=2));save()
cmd=[sys.executable,str(ROOT/'scripts/train_registered_task.py'),'--task',state['task'],'--num_envs',str(args.num_envs),'--max_iterations',str(args.iterations),'--seed',str(args.seed),'--run_name',name,'--headless']
state['training_command']=cmd
trainlog=(logs/'train.log').open('w');train=subprocess.Popen(cmd,cwd=ROOT,stdout=trainlog,stderr=subprocess.STDOUT);state['training_pid']=train.pid;state['status']='training';save()
worker=None;workerfile=None;pending=[];seen=set();milestones=sorted(set(n for n in [500,1000,3000,10000,args.iterations-1] if n<args.iterations));last_sizes={}
try:
 while True:
  dirs=list((ROOT/'logs/rsl_rl/beyondmimic_dance').glob('*_'+name))
  if dirs:
   run=dirs[0];state['run']=str(run)
   checkpoints=list(run.glob('model_*.pt'));state['latest_checkpoint_iteration']=max([int(c.stem.split('_')[-1]) for c in checkpoints],default=-1)
   for it in milestones:
    cp=run/f'model_{it}.pt'
    if cp.exists() and it not in seen:
     size=cp.stat().st_size
     if size and last_sizes.get(it)==size:
      seen.add(it);pending.append((it,'video',cp))
      if it in [1000,3000,10000,args.iterations-1]:pending.append((it,'audit',cp))
     last_sizes[it]=size
  if worker and worker.poll() is not None:
   workerfile.close();it,kind,cp,out=active;entry=state['milestones'].setdefault(str(it),{});entry[kind+'_exit_code']=worker.returncode
   if worker.returncode==0 and (out/'evaluation.json').exists():
    evaluation=json.loads((out/'evaluation.json').read_text());entry[kind]=str(out/'evaluation.json');entry['checkpoint_sha256']=sha(cp)
    if kind=='video':
     subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(out/'demo.mp4')],check=True,stdout=(out/'ffprobe.json').open('w'))
     entry['video_sha256']=sha(out/'demo.mp4')
    else:
     entry['gate']=evaluation['gate']
     if it==args.iterations-1:state['task_gate_passed']=evaluation['gate']['passed']
   else:entry[kind+'_error']='See evaluation log; failed evaluations are not successful milestones'
   worker=None;save()
  if not worker and pending:
   it,kind,cp=pending.pop(0);out=session/f'iteration_{it}'/kind;out.parent.mkdir(exist_ok=True)
   command=[sys.executable,str(ROOT/'scripts/beyondmimic/evaluate.py'),'--task',state['task'],'--skill','dance','--checkpoint',str(cp),'--output',str(out),'--num_envs','1' if kind=='video' else '64','--steps','500' if kind=='video' else '1455','--headless']
   if kind=='video':command.append('--video')
   workerfile=(logs/f'{it}.{kind}.log').open('w');worker=subprocess.Popen(command,cwd=ROOT,stdout=workerfile,stderr=subprocess.STDOUT);active=(it,kind,cp,out);state['evaluation_pid']=worker.pid;save()
  code=train.poll()
  if code is not None:
   state['training_exit_code']=code;state['training_completed']=code==0 and state.get('latest_checkpoint_iteration',-1)>=args.iterations-1
   if code!=0:raise RuntimeError(f'Training exited {code}; see {logs}/train.log')
   if not worker and not pending and all(it in seen for it in milestones):break
  save();time.sleep(10)
 state['status']='completed_budget_and_evaluations';save()
except BaseException as e:
 state['status']='failed';state['error']=repr(e);save();traceback.print_exc()
 # Only child processes created by this supervisor, never unrelated workloads.
 if train.poll() is None:train.terminate()
 if worker and worker.poll() is None:worker.terminate()
 raise
finally:trainlog.close()
