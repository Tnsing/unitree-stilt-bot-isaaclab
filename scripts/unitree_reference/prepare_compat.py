#!/usr/bin/env python3
"""Reproducible, explicit PhysX backport of the pinned Unitree 3.0 source.

The untouched snapshot remains under third_party; no global Isaac installation is changed.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab'
DEST = ROOT / 'vendor/unitree_rl_lab'


def transform(path, text):
    text = text.replace('from isaaclab_physx.physics import PhysxCfg', 'from isaaclab.sim import PhysxCfg')
    text = text.replace('import warp as wp', 'from unitree_rl_lab.compat_tensor import tensor as as_torch\nimport warp as wp')
    text = text.replace('wp.to_torch(', 'as_torch(')
    if path == 'tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py':
        tree = ast.parse(text)
        removed_classes = {'PhysicsCfg', 'ContactSensorPresetCfg', 'NewtonEventCfg', 'EventCfgPreset'}
        cuts = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module.startswith(('isaaclab_newton', 'isaaclab_physx')) or node.module == 'isaaclab_tasks.utils'):
                cuts.append((node.lineno, node.end_lineno))
            if isinstance(node, ast.ClassDef):
                if node.name in removed_classes:
                    cuts.append((min([d.lineno for d in node.decorator_list]+[node.lineno]), node.end_lineno))
                elif node.name == 'RobotEnvCfg':
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and method.name == 'validate_config':
                            cuts.append((method.lineno, method.end_lineno))
        lines = text.splitlines(keepends=True)
        for start,end in sorted(cuts,reverse=True):del lines[start-1:end]
        text = ''.join(lines)
        text = text.replace('from isaaclab.sensors import RayCasterCfg, patterns','from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns')
        text = text.replace('contact_forces: ContactSensorPresetCfg = ContactSensorPresetCfg()', 'contact_forces: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)')
        text = text.replace('SimulationCfg(physics=PhysicsCfg())','SimulationCfg(physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))')
        text = text.replace('events: EventCfgPreset = EventCfgPreset()', 'events: PhysxEventCfg = PhysxEventCfg()')
        text = text.replace('self.scene.contact_forces.physx.update_period', 'self.scene.contact_forces.update_period')
        text = text.replace('        self.scene.contact_forces.newton.update_period = self.sim.dt\n','')
    if path.endswith('/tracking_env_cfg.py'):
        text = text.replace('self.sim.physics = PhysxCfg(', 'self.sim.physx = PhysxCfg(')
    if path == 'tasks/mimic/mdp/commands.py':
        text = text.replace('self.body_quat_relative_w[:, :, 3] = 1.0  # XYZW format: w is at index 3', 'self.body_quat_relative_w[:, :, 0] = 1.0  # Isaac Lab 2.3 uses WXYZ')
        for kind,arg in [('joint_position','position'),('joint_velocity','velocity'),('root_pose','root_pose'),('root_velocity','root_velocity')]:
            text = text.replace(f'write_{kind}_to_sim_index',f'write_{kind}_to_sim')
    if path == 'tasks/mimic/mdp/events.py':
        # Official 2.3 COM API takes torch tensors, not 3.0 Warp views.
        tree=ast.parse(text)
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='randomize_rigid_body_com')
        text=''.join(text.splitlines(keepends=True)[:n.lineno-1])+'from isaaclab.envs.mdp import randomize_rigid_body_com\n'
    if path.endswith('agents/rsl_rl_ppo_cfg.py'):
        text=text.replace('"actor": ["policy"]','"policy": ["policy"]')
    # Avoid auto-import of unsupported Newton/direct/warp tasks and registration side effects.
    if path.endswith('__init__.py') and '/mdp/' not in '/'+path:
        text='"""Pinned Unitree package; project registers explicitly selected PhysX tasks."""\n'
    return text


def main():
    changes=[]
    for p in SOURCE.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(SOURCE).as_posix()
        if '/direct/' in rel or rel.endswith(('velocity_direct_env_cfg.py','velocity_warp_env_cfg.py')):continue
        if p.suffix not in ('.py','.csv'):continue
        out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True)
        original=p.read_text(); adapted=transform(rel,original) if p.suffix=='.py' else original
        if p.suffix=='.py':ast.parse(adapted)
        out.write_text(adapted)
        if adapted!=original:changes.append(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile='upstream/'+rel,tofile='compat/'+rel)))
    (DEST/'compat_tensor.py').write_text('import torch\nimport warp as wp\n\ndef tensor(value):\n    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)\n')
    out=ROOT/'artifacts/upstream/unitree_isaaclab23.patch';out.write_text('\n'.join(changes))
    print(json.dumps({'source_commit':'2522f89ab8ad6b1188b0368d122d16cb0e3cb8c8','runtime':'Isaac Lab 2.3 / Isaac Sim 5.1 / PhysX','patch':str(out),'patch_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'vendor':str(DEST)},indent=2))

if __name__=='__main__':main()
