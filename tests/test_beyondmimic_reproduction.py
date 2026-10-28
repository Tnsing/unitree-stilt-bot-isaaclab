import hashlib,importlib.util,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reset_sync',ROOT/'src/stilt_bot/tasks/beyondmimic/reset_sync.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Motion:
    def __init__(self,frames,increment=1):self.time_steps=torch.tensor(frames);self.increment=increment;self.cache=None
    def _update_command(self):self.time_steps+=self.increment;self.cache=self.time_steps.clone()

class ReproductionTests(unittest.TestCase):
    def test_reset_cache_preserves_reference_frame(self):
        motion=Motion([0,37,1456]);mod.refresh_motion_cache(motion)
        self.assertEqual(motion.time_steps.tolist(),[0,37,1456]);self.assertEqual(motion.cache.tolist(),[0,37,1456])
    def test_reset_cache_detects_accidental_advance(self):
        with self.assertRaises(RuntimeError):mod.refresh_motion_cache(Motion([0,1],increment=2))
    def test_real_unprimed_reset_negative_control(self):
        j=json.loads((ROOT/'artifacts/beyondmimic/legacy_dance_reset_regression_unprimed/evaluation.json').read_text())
        self.assertEqual(j['task_termination_rate'],1.0)
        self.assertEqual(set(j['first_failure_step']),{0})
        self.assertEqual(j['termination_causes']['ee_body_pos'],64)
    def test_milestone_video_is_not_full_clip_evidence(self):
        path=ROOT/'artifacts/beyondmimic/formal_official_mdp_seed42_2026-10-29_00-33-12/iteration_500/video'
        j=json.loads((path/'evaluation.json').read_text())
        self.assertFalse(j['horizon_complete']);self.assertFalse(j['gate']['passed'])
        probe=json.loads((path/'ffprobe.json').read_text())['streams'][0]
        self.assertEqual((probe['width'],probe['height']),(1280,720));self.assertEqual(int(probe['nb_read_frames']),500)
    def test_original_source_hashes(self):
        lock=json.loads((ROOT/'configs/beyondmimic_lock.json').read_text())
        base=ROOT/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking'
        for rel,sha in lock['source_sha256'].items():
            self.assertEqual(hashlib.sha256((base/rel).read_bytes()).hexdigest(),sha,rel)
    def test_compatible_sources_have_only_declared_changes(self):
        lock=json.loads((ROOT/'configs/beyondmimic_lock.json').read_text())
        for rel in lock['source_sha256']:
            expected=(ROOT/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking'/rel).read_text().replace('AdditiveUniformNoiseCfg','UniformNoiseCfg').replace('math_utils.quat_rotate_inverse(', 'math_utils.quat_apply_inverse(')
            if rel in ['__init__.py','tasks/__init__.py','tasks/tracking/config/g1/__init__.py','tasks/tracking/config/humanoid/__init__.py']:expected='"""Pinned official source; registration is project-owned."""\n'
            self.assertEqual((ROOT/'vendor/whole_body_tracking'/rel).read_text(),expected,rel)
    def test_corrected_legacy_audit_not_first_step_failure(self):
        j=json.loads((ROOT/'artifacts/beyondmimic/legacy_dance_reset_regression_fixed/evaluation.json').read_text())
        self.assertEqual(j['reset_diagnostics']['before_refresh']['ee_body_pos'],64)
        self.assertEqual(j['reset_diagnostics']['after_refresh']['ee_body_pos'],0)
        self.assertEqual(j['task_termination_rate'],0)
        self.assertTrue(j['gate']['passed'])

if __name__=='__main__':unittest.main()
