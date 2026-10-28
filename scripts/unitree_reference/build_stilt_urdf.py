#!/usr/bin/env python3
"""Build a physical 0.30 m rigid stilt pair with analytic mass/inertia.

Prototype dimensions, not a reconstruction of any real hardware.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'assets/unitree_ros/g1_description'
tree=ET.parse(folder/'g1_29dof_rev_1_0.urdf');robot=tree.getroot()
length=.30; radius=.018; pole_mass=.60; pad_mass=.20
sx,sy,sz=.12,.08,.02
# Link frame is the ankle frame; each stilt's COM is determined from pole and pad.
pole_z=-.035-length/2; pad_z=-.035-length-sz/2
mass=pole_mass+pad_mass; com_z=(pole_mass*pole_z+pad_mass*pad_z)/mass
ixy=pole_mass*(3*radius**2+length**2)/12
inertia=[ixy+pole_mass*(pole_z-com_z)**2+pad_mass*(sy**2+sz**2)/12+pad_mass*(pad_z-com_z)**2,
         ixy+pole_mass*(pole_z-com_z)**2+pad_mass*(sx**2+sz**2)/12+pad_mass*(pad_z-com_z)**2,
         pole_mass*radius**2/2+pad_mass*(sx**2+sy**2)/12]
for side in ('left','right'):
 link=ET.SubElement(robot,'link',name=f'{side}_stilt_link')
 inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'origin',xyz=f'.035 0 {com_z}',rpy='0 0 0');ET.SubElement(inertial,'mass',value=str(mass));ET.SubElement(inertial,'inertia',ixx=str(inertia[0]),iyy=str(inertia[1]),izz=str(inertia[2]),ixy='0',ixz='0',iyz='0')
 for tag in ('visual','collision'):
  el=ET.SubElement(link,tag);ET.SubElement(el,'origin',xyz=f'.035 0 {pole_z}',rpy='0 0 0');geom=ET.SubElement(el,'geometry');ET.SubElement(geom,'cylinder',radius=str(radius),length=str(length))
  if tag=='visual':ET.SubElement(ET.SubElement(el,'material',name='stilt_blue'),'color',rgba='.05 .25 .85 1')
  el=ET.SubElement(link,tag);ET.SubElement(el,'origin',xyz=f'.035 0 {pad_z}',rpy='0 0 0');geom=ET.SubElement(el,'geometry');ET.SubElement(geom,'box',size=f'{sx} {sy} {sz}')
  if tag=='visual':ET.SubElement(ET.SubElement(el,'material',name='stilt_pad'),'color',rgba='.08 .08 .08 1')
 joint=ET.SubElement(robot,'joint',name=f'{side}_stilt_fixed',type='fixed');ET.SubElement(joint,'parent',link=f'{side}_ankle_roll_link');ET.SubElement(joint,'child',link=f'{side}_stilt_link');ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0')
ET.indent(tree);out=folder/'g1_29dof_stilt_030.urdf';tree.write(out,encoding='utf-8',xml_declaration=True)
spec={'description':'Parameterized simulated rigid-stilt prototype; not verified hardware geometry','pole_length_m':length,'pole_radius_m':radius,'pad_xyz_m':[sx,sy,sz],'mass_per_stilt_kg':mass,'com_local_m':[.035,0,com_z],'diagonal_inertia_kg_m2':inertia,'total_added_mass_kg':2*mass,'fixed_joints':['left_stilt_fixed','right_stilt_fixed'],'merge_fixed_joints':False,'extra_actuators':0,'sole_drop_relative_to_original_m':length+sz,'urdf':str(out)}
(folder/'stilt_spec.json').write_text(json.dumps(spec,indent=2));print(json.dumps(spec,indent=2))
