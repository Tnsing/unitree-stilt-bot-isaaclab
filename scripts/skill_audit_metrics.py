"""Pure NumPy physical gates; training reward and absence of exceptions are not gates."""
import numpy as np


def _mean(a, mask):
    v=a[mask]
    return float(v.mean()) if v.size else None


def summarize(d,skill,dt,command_speed):
    valid=d['valid'].astype(bool)
    n=valid.shape[1]
    settled=valid & (np.arange(len(valid))[:,None]*dt >= 1.0)
    speed=d['velocity'][:,:,0]
    tilt=np.arccos(np.clip(-d['gravity'][:,:,2],-1,1))
    force=np.linalg.norm(d['force'],axis=-1)
    feet_contact=force>5.0
    air=~feet_contact.any(axis=-1)
    terminated=d['terminated'].any(axis=0)
    result={'physical_fall_rate':float(terminated.mean()),'full_horizon_survival_rate':float((~terminated).mean()),
        'mean_forward_speed_mps':_mean(speed,settled),'forward_rmse_mps':None,
        'mean_lateral_speed_abs_mps':_mean(abs(d['velocity'][:,:,1]),settled),
        'mean_yaw_speed_abs_radps':_mean(abs(d['angular_velocity'][:,:,2]),settled),
        'p95_tilt_degrees':float(np.percentile(tilt[valid],95)*180/np.pi) if valid.any() else None,
        'both_feet_unloaded_fraction':_mean(air,settled),
        'mean_root_height_m':_mean(d['position'][:,:,2],settled)}
    if settled.any():result['forward_rmse_mps']=float(np.sqrt(np.mean((speed[settled]-command_speed)**2)))
    # Sustained, force-backed takeoff followed by landing with actual foot clearance.
    jumps=np.zeros(n,dtype=int); flight_max=np.zeros(n); running_flights=np.zeros(n,dtype=int)
    for j in range(n):
        start=None
        for i in range(len(valid)):
            if not valid[i,j]:
                start=None;continue
            if air[i,j] and start is None:start=i
            if not air[i,j] and start is not None:
                duration=(i-start)*dt
                foot_clearance=float(d['feet_position'][start:i,j,:,2].min(axis=1).max())
                if duration>=.06:
                    running_flights[j]+=1
                    flight_max[j]=max(flight_max[j],duration)
                if duration>=.10 and foot_clearance>.06:
                    jumps[j]+=1
                start=None
    result['completed_jump_count_by_env']=jumps.tolist()
    result['flight_event_count_by_env']=running_flights.tolist()
    result['max_completed_flight_duration_s']=flight_max.tolist()
    if skill in ['walk','run']:
        result['skill_gate']={'definition':'survival >=95%, forward RMSE <=0.25 m/s (walk) or <=0.4 (run); run additionally requires repeated force-backed flight',
          'passed':bool((~terminated).mean()>=.95 and result['forward_rmse_mps'] is not None and result['forward_rmse_mps']<=(.25 if skill=='walk' else .4) and (skill!='run' or (running_flights>=3).mean()>=.8))}
    elif skill=='jump':
        result['skill_gate']={'definition':'survival >=95%, >=3 completed 0.10s flight+landing events with both feet height >0.06m in >=80% envs; lateral drift <=0.4m',
          'passed':bool((~terminated).mean()>=.95 and (jumps>=3).mean()>=.8 and _mean(np.linalg.norm(d['position'][:,:,:2],axis=-1),settled) is not None and _mean(np.linalg.norm(d['position'][:,:,:2],axis=-1),settled)<=.4)}
    elif skill=='dance':
        period=2.4;phase=2*np.pi*d['phase_time']/period
        target_vy=.45*np.sin(phase); target_wz=.8*np.sin(2*phase)
        vy_rmse=float(np.sqrt(np.mean((d['velocity'][:,:,1][settled]-target_vy[settled])**2))) if settled.any() else None
        wz_rmse=float(np.sqrt(np.mean((d['angular_velocity'][:,:,2][settled]-target_wz[settled])**2))) if settled.any() else None
        arm_sign=np.ones(d['arm_position'].shape[-1]);arm_sign[1::2]=-1
        target=.35*np.sin(phase)[:,:,None]*arm_sign
        arm_rmse=float(np.sqrt(np.mean((d['arm_position'][settled]-target[settled])**2))) if settled.any() else None
        result.update(dance_vy_rmse_mps=vy_rmse,dance_wz_rmse_radps=wz_rmse,dance_arm_rmse_rad=arm_rmse)
        result['skill_gate']={'definition':'survival >=95%, phase-aligned body velocity RMSE <=0.25m/s and yaw RMSE <=0.35rad/s, arm RMSE <=0.2rad',
          'passed':bool((~terminated).mean()>=.95 and vy_rmse is not None and vy_rmse<=.25 and wz_rmse<=.35 and arm_rmse<=.2)}
    result['evidence_boundary']='Held-out simulator rollout. Not hardware validation; no success inference from reward.'
    return result
