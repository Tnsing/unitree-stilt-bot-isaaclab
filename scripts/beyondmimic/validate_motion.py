#!/usr/bin/env python3
"""CPU numerical/reference-schema checks (not a learned-policy evaluation)."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'assets/motions/beyondmimic_dance_102_50hz.json').read_text())
path=Path(manifest['output']);data=np.load(path)
raw=np.loadtxt(manifest['source_csv'],delimiter=',')
t=np.arange(len(data['joint_pos']))/50
interpolated=np.stack([np.interp(t,np.arange(len(raw))/60,raw[:,7+i]) for i in range(29)],axis=-1)
order=[manifest['joint_names'].index(n) for n in manifest['sdk_joint_order']]
error=float(np.max(np.abs(data['joint_pos'][:,order]-interpolated)))
quat_error=float(np.max(np.abs(np.linalg.norm(data['body_quat_w'],axis=-1)-1)))
fd=np.gradient(data['body_pos_w'],.02,axis=0)
vel_rmse=float(np.sqrt(np.mean((data['body_lin_vel_w'][2:-2]-fd[2:-2])**2)))
result={'reference_only':True,'sha256_verified':hashlib.sha256(path.read_bytes()).hexdigest()==manifest['output_sha256'],'frames':len(t),'fps':50,'num_joints':len(manifest['joint_names']),'num_bodies':len(manifest['body_names']),'joint_mapping_max_abs_error_rad':error,'body_quaternion_max_norm_error':quat_error,'body_lin_vel_vs_finite_difference_rmse_mps':vel_rmse,'all_finite':all(np.isfinite(data[k]).all() for k in data.files),'passed':bool(error<1e-3 and quat_error<1e-5 and vel_rmse<.2)}
(ROOT/'artifacts/beyondmimic/motion_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed'] and result['sha256_verified']
