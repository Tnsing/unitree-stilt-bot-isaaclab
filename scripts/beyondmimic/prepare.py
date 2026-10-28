#!/usr/bin/env python3
"""Prepare pinned, isolated official BeyondMimic sources; do not edit Isaac Lab."""
import difflib, hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
COMMIT = '9db915838b17785c6fec90c91fe80b95b147e1bd'
source = ROOT / 'third_party/whole_body_tracking'
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
pkg = source / 'source/whole_body_tracking/whole_body_tracking'
out = ROOT / 'vendor/whole_body_tracking'
archive = ROOT / 'third_party/unitree_description.tar.gz'
assert archive.exists(), 'Download the official GCS robot description archive first'
lock_path=ROOT/'configs/beyondmimic_lock.json'
previous_lock=json.loads(lock_path.read_text()) if lock_path.exists() else None
if previous_lock:
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==previous_lock['robot_archive_sha256'], 'Robot archive SHA mismatch'
source_hashes = {}
patches = []
for p in pkg.rglob('*.py'):
    rel = p.relative_to(pkg)
    text = p.read_text()
    source_hashes[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    fixed = text.replace('AdditiveUniformNoiseCfg', 'UniformNoiseCfg').replace('math_utils.quat_rotate_inverse(', 'math_utils.quat_apply_inverse(')
    # Register only the project-owned tasks; avoid recursive optional SMPL imports.
    if str(rel) in ['__init__.py', 'tasks/__init__.py', 'tasks/tracking/config/g1/__init__.py', 'tasks/tracking/config/humanoid/__init__.py']:
        fixed = '"""Pinned official source; registration is project-owned."""\n'
    if previous_lock:
        assert source_hashes[str(rel)]==previous_lock['source_sha256'][str(rel)], 'Pinned source SHA mismatch: '+str(rel)
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fixed)
    patches.extend(difflib.unified_diff(text.splitlines(True), fixed.splitlines(True), fromfile='a/'+str(rel), tofile='b/'+str(rel)))
with tarfile.open(archive) as tar:
    base = (out/'assets').resolve()
    for member in tar.getmembers():
        target = (base/member.name).resolve()
        assert target == base or base in target.parents, 'Unsafe archive path'
        assert member.isfile() or member.isdir(), 'Archive links/devices are not allowed'
    tar.extractall(base)
assert (out/'assets/unitree_description/urdf/g1/main.urdf').exists()
assets = {str(p.relative_to(out/'assets')): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'assets').rglob('*') if p.is_file()}
lock = {'repository':'https://github.com/HybridRobotics/whole_body_tracking', 'commit':COMMIT,
        'source_sha256':source_hashes, 'robot_archive_url':'https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz?generation=1749613070556521',
        'robot_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'asset_sha256':assets,
        'runtime':'Isaac Lab 2.3.0 / Isaac Sim 5.1.0 / PhysX',
        'boundary':'Official tracking MDP and G1 cylinder asset, API compatibility port; not guided diffusion, mjlab, or original runtime exact reproduction.'}
if previous_lock:
    assert lock==previous_lock, 'Prepared source/assets differ from committed lock'
else:
    lock_path.write_text(json.dumps(lock,indent=2))
(ROOT/'artifacts/upstream/beyondmimic_lab23.patch').write_text(''.join(patches))
shutil.copyfile(source/'LICENCE',ROOT/'artifacts/upstream/beyondmimic.LICENCE')
print('Prepared', COMMIT, len(source_hashes), 'Python sources;', len(assets), 'asset files')
