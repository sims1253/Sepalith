import unittest
from azure_container_state import current_state


class StateTests(unittest.TestCase):
    def test_accepted_deployment_has_no_runtime_view_yet(self):
        for resource in ({}, {'containers': None}, {'containers': []},
                         {'containers': [{}]}, {'containers': [{'instanceView': None}]},
                         {'containers': [{'instanceView': {'currentState': None}}]}):
            with self.subTest(resource=resource):
                self.assertEqual(current_state(resource), {})

    def test_terminal_exit_code_is_preserved(self):
        state = {'state': 'Terminated', 'exitCode': 3}
        self.assertEqual(current_state({'containers': [{'instanceView': {'currentState': state}}]}), state)


if __name__ == '__main__':
    unittest.main()
