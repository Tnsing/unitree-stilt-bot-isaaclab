"""Same-frame reference cache refresh, never bypassing a termination condition."""
def refresh_motion_cache(motion):
    before = motion.time_steps.clone()
    motion.time_steps -= 1
    motion._update_command()
    if not (motion.time_steps == before).all():
        raise RuntimeError('Reference refresh unexpectedly advanced the motion')
