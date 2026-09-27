import unittest
from host_memory_policy import HostMemoryPolicy

def observation(free=16000, committed=70, limit=100, events=None):
    return dict(AvailableMBytes=free, CommittedBytes=committed,
                CommitLimit=limit, DriverEvents=events or [])

class GuardPolicyTests(unittest.TestCase):
    def test_single_soft_dip_recovers(self):
        p = HostMemoryPolicy()
        self.assertIsNone(p.reason(observation(7198)))
        self.assertIsNone(p.reason(observation(15592)))
        self.assertIsNone(p.reason(observation(7198)))

    def test_two_low_samples_stop(self):
        p = HostMemoryPolicy()
        self.assertIsNone(p.reason(observation(7198)))
        self.assertEqual(p.reason(observation(7500)), 'host_free_memory_below_soft_floor_persisted')

    def test_hard_floor_stops_immediately(self):
        self.assertEqual(HostMemoryPolicy().reason(observation(4000)), 'host_free_memory_below_hard_floor')

    def test_admission_requires_soft_floor(self):
        self.assertEqual(HostMemoryPolicy().reason(observation(7198), preflight=True), 'host_free_memory_below_admission_floor')
        self.assertIsNone(HostMemoryPolicy().reason(observation(8192), preflight=True))

    def test_driver_and_commit_override_hysteresis(self):
        self.assertEqual(HostMemoryPolicy().reason(observation(events=[{'Id':153}])), 'new_nvidia_driver_event')
        self.assertEqual(HostMemoryPolicy().reason(observation(committed=91)), 'host_commit_above90percent')

    def test_invalid_thresholds_fail(self):
        for args in [(4096,8192,2),(8192,0,2),(8192,4096,0)]:
            with self.assertRaises(ValueError): HostMemoryPolicy(*args)

if __name__ == '__main__':
    unittest.main()
