"""Bounded host-memory hysteresis for one supervised campaign workload."""
class HostMemoryPolicy:
    def __init__(self, soft_mib=8192, hard_mib=4096, samples=2):
        if not (0 < hard_mib < soft_mib) or samples < 1:
            raise ValueError('invalid memory thresholds')
        self.soft_mib = soft_mib
        self.hard_mib = hard_mib
        self.samples = samples
        self.below_soft = 0

    def reason(self, value, *, preflight=False):
        if value['DriverEvents']:
            return 'new_nvidia_driver_event'
        if value['CommittedBytes'] / value['CommitLimit'] > 0.9:
            return 'host_commit_above90percent'
        free = value['AvailableMBytes']
        if free < self.hard_mib:
            return 'host_free_memory_below_hard_floor'
        if preflight:
            return 'host_free_memory_below_admission_floor' if free < self.soft_mib else None
        self.below_soft = self.below_soft + 1 if free < self.soft_mib else 0
        if self.below_soft >= self.samples:
            return 'host_free_memory_below_soft_floor_persisted'
        return None
