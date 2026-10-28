#!/usr/bin/env python3
"""Write evidence from real files; training-complete and task-passed are separate."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--budget',type=int,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[2];run=args.run.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect(path):
 if not path.is_file():raise FileNotFoundError(path)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
checkpoints=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]));assert checkpoints
scalars={};events=sorted(run.glob('events.out.tfevents.*'));assert events
for event in events:
 acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload()
 for tag in acc.Tags()['scalars']:
  values=acc.Scalars(tag)
  if values:scalars[tag]={'step':int(values[-1].step),'value':float(values[-1].value)}
final_step=int(checkpoints[-1].stem.split('_')[-1]);complete=final_step>=args.budget-1 and scalars.get('Train/mean_reward',{}).get('step',-1)>=args.budget-1
artefacts=root/'artifacts/unitree_reference'/args.skill
video_evals=[]
for evaluation in sorted(artefacts.glob('video_*/evaluation.json')):
 d=json.loads(evaluation.read_text());v=Path(d['video']);cp=Path(d['checkpoint']);assert sha(cp)==d['checkpoint_sha256'];assert sha(v)==d['video_sha256']
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames,r_frame_rate,duration','-of','json',str(v)],text=True))
 video_evals.append({'evaluation':str(evaluation),'checkpoint':inspect(cp),'video':inspect(v),'metadata':probe})
audit_path=artefacts/'audit_final/evaluation.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
if audit:assert audit['checkpoint_sha256']==sha(checkpoints[-1])
manifest={'created_at':datetime.now(timezone.utc).isoformat(),'skill':args.skill,'training_completed':complete,'task_gate_passed':audit['gate']['passed'] if audit else None,'declared_iteration_budget':args.budget,'final_iteration':final_step,'run_dir':str(run),'configs':[inspect(run/'params/env.yaml'),inspect(run/'params/agent.yaml')],'events':[inspect(e) for e in events],'checkpoints':[dict(iteration=int(c.stem.split('_')[-1]),**inspect(c)) for c in checkpoints],'last_scalars':scalars,'milestone_videos':video_evals,'audit':str(audit_path) if audit else None,'upstream':json.loads((root/'third_party/unitree_rl_lab/UPSTREAM.json').read_text()),'compat_patch':inspect(root/'artifacts/upstream/unitree_isaaclab23.patch'),'asset_provenance':json.loads((root/'assets/unitree_ros/g1_description/PROVENANCE.json').read_text()),'stilt_geometry':args.skill=='stilt','stilt_validation':str(root/'artifacts/unitree_reference/stilt_physics/validation.json') if args.skill=='stilt' else None,'git_revision_at_collection':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_files_at_collection':{str(x.relative_to(root)):sha(x) for folder in ['src/stilt_bot/tasks/unitree_reference','scripts/unitree_reference'] for x in (root/folder).rglob('*') if x.is_file() and '__pycache__' not in str(x)},'boundary':'Training budget completion is separate from convergence, skill gate success and real-hardware validation.'}
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(manifest,indent=2));print(json.dumps({'manifest':str(args.out),'training_completed':complete,'task_gate_passed':manifest['task_gate_passed'],'videos':len(video_evals)}))
