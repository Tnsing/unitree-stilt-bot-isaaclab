"""Observable, contact-conditioned references for V1 skill experiments.

These synthetic references are experiment design, not mocap reproduction.
"""
import math
import torch
from isaaclab.managers import SceneEntityCfg


def phase(env,period):
    return torch.remainder(env.episode_length_buf.float()*env.step_dt,period)/period


def clock(env,period):
    p=2*math.pi*phase(env,period)
    return torch.stack([p.sin(),p.cos(),(2*p).sin(),(2*p).cos()],dim=-1)


def root_local(env):
    return env.scene['robot'].data.root_pos_w-env.scene.env_origins


def foot_contact(env,sensor_cfg):
    return env.scene.sensors[sensor_cfg.name].data.net_forces_w[:,sensor_cfg.body_ids,:].norm(dim=-1)>5.


def contact_observation(env,sensor_cfg):
    return foot_contact(env,sensor_cfg).float()


def horizontal_stability(env):
    r=env.scene['robot'].data
    return r.root_lin_vel_w[:,:2].square().sum(-1)+.2*r.root_ang_vel_w[:,2].square()


def local_drift(env):
    return root_local(env)[:,:2].square().sum(-1)


def height_target(env,period):
    """Stand, crouch, launch, physical ballistic flight and controlled landing."""
    p=phase(env,period)
    h=torch.full_like(p,.74);v=torch.zeros_like(p)
    # Crouch 0.08..0.25, push off 0.25..0.32. Flight 0.32..0.62.
    m=(p>=.08)&(p<.25);u=(p-.08)/.17
    h=torch.where(m,.74-.10*.5*(1-torch.cos(math.pi*u)),h)
    v=torch.where(m,-.05*math.pi/(.17*period)*torch.sin(math.pi*u),v)
    m=(p>=.25)&(p<.32);u=(p-.25)/.07
    h=torch.where(m,.64+.10*u,h)
    v=torch.where(m,torch.full_like(p,.10/(.07*period)),v)
    t=(p-.32)*period;T=.30*period;vz=.5*9.81*T
    m=(p>=.32)&(p<.62)
    h=torch.where(m,.74+vz*t-.5*9.81*t.square(),h)
    v=torch.where(m,vz-9.81*t,v)
    m=(p>=.62)&(p<.82);u=(p-.62)/.20
    h=torch.where(m,.74-.07*torch.sin(math.pi*u),h)
    v=torch.where(m,-.07*math.pi/(.20*period)*torch.cos(math.pi*u),v)
    return h,v


def jump_reference(env,period):
    h,v=height_target(env,period)
    return torch.stack([h,v],dim=-1)


def jump_tracking(env,period):
    h,v=height_target(env,period);r=env.scene['robot'].data
    return torch.exp(-((root_local(env)[:,2]-h)/.09).square()-.3*((r.root_lin_vel_w[:,2]-v)/.8).square())


def jump_contact_reward(env,period,sensor_cfg):
    p=phase(env,period);flight=(p>=.35)&(p<=.59)
    feet=foot_contact(env,sensor_cfg)
    score=torch.where(flight,(~feet.any(-1)).float(),feet.all(-1).float())
    upright=(-env.scene['robot'].data.projected_gravity_b[:,2]).clamp(0,1)
    return score*upright


def run_contact_reward(env,period,sensor_cfg):
    p=phase(env,period)
    foot_phase=torch.stack([p,torch.remainder(p+.5,1.)],dim=-1)
    expected=foot_phase<.38
    actual=foot_contact(env,sensor_cfg)
    upright=(-env.scene['robot'].data.projected_gravity_b[:,2]).clamp(0,1)
    return (actual==expected).float().mean(-1)*upright


def run_flight_reward(env,period,sensor_cfg,asset_cfg):
    p=phase(env,period)
    flight=((p>=.38)&(p<.5))|(p>=.88)
    contact=foot_contact(env,sensor_cfg)
    z=env.scene['robot'].data.body_pos_w[:,asset_cfg.body_ids,2]-env.scene.env_origins[:,None,2]
    return flight.float()*(~contact.any(-1)).float()*(z.min(-1).values>.07).float()


def dance_targets(env,period):
    p=2*math.pi*phase(env,period)
    # Bounded center-of-mass path and yaw rhythm, with explicit phase supplied.
    return torch.stack([.45*p.sin(),.8*(2*p).sin()],dim=-1)
