import unittest
from collections import deque

from controller import PressEdge, Switch2Reader, rear_buttons


def report(button_byte=0, other_byte=0):
    data = [0] * 64
    data[0], data[8], data[5] = 5, button_byte, other_byte
    return data


class FakeDevice:
    def __init__(self):
        self.reports = deque()
        self.closed = False

    def open_path(self, path):
        self.closed = False

    def set_nonblocking(self, value):
        pass

    def read(self, size):
        value = self.reports.popleft() if self.reports else []
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self):
        self.pad = FakeDevice()
        self.devices = [dict(path=b'test-pad', usage_page=1, usage=5, interface_number=0)]

    def enumerate(self, vendor, product):
        assert (vendor, product) == (0x057e, 0x2069)
        return self.devices

    def device(self):
        return self.pad


class ControllerTests(unittest.TestCase):
    def test_physical_report_gr_is_distinct_from_gl_and_game_buttons(self):
        self.assertEqual(rear_buttons(report(1)), (True, False))
        self.assertEqual(rear_buttons(report(2)), (False, True))
        self.assertEqual(rear_buttons(report(0, 255)), (False, False))
        self.assertEqual(rear_buttons(report(0xF0)), (False, False))

    def test_unknown_or_truncated_reports_cannot_trigger(self):
        self.assertIsNone(rear_buttons([5] * 9))
        value = report(1)
        value[0] = 1
        self.assertIsNone(rear_buttons(value))
        self.assertIsNone(rear_buttons([]))

    def test_initial_hold_and_long_hold_do_not_repeat(self):
        edge = PressEdge()
        self.assertFalse(edge.update(True, 0))
        self.assertFalse(edge.update(True, 1))
        self.assertFalse(edge.update(False, 2))
        self.assertTrue(edge.update(True, 3))
        self.assertFalse(edge.update(True, 10))
        self.assertFalse(edge.update(False, 11))
        self.assertTrue(edge.update(True, 12))

    def test_bounce_guard_requires_fresh_release_and_press(self):
        edge = PressEdge()
        edge.update(False, 0)
        self.assertTrue(edge.update(True, 1))
        edge.update(False, 1.05)
        self.assertFalse(edge.update(True, 1.1))
        self.assertFalse(edge.update(True, 2))
        edge.update(False, 3)
        self.assertTrue(edge.update(True, 3.1))

    def test_release_bounce_after_long_hold_cannot_trigger_again(self):
        edge = PressEdge()
        edge.update(False, 0)
        self.assertTrue(edge.update(True, 1))
        self.assertFalse(edge.update(True, 10))
        self.assertFalse(edge.update(False, 10.001))
        self.assertFalse(edge.update(True, 10.006))
        self.assertFalse(edge.update(True, 11))
        edge.update(False, 12)
        self.assertTrue(edge.update(True, 12.05))

    def setUp(self):
        self.backend = FakeBackend()
        self.reader = Switch2Reader(self.backend)

    def poll(self, now, *reports, enabled=True, button='GR'):
        self.backend.pad.reports.extend(reports)
        return self.reader.poll(enabled, button, now)

    def test_raw_reports_call_only_selected_rear_button_once(self):
        self.assertFalse(self.poll(0, report()))
        self.assertFalse(self.poll(1, report(2), report()))
        self.assertTrue(self.poll(2, report(1), report(1), report()))
        self.assertFalse(self.poll(3))
        self.assertTrue(self.poll(4, report(1)))

    def test_mapping_change_requires_release_before_trigger(self):
        self.poll(0, report())
        self.assertFalse(self.poll(1, report(2), button='GL'))
        self.assertFalse(self.poll(2, report(), button='GL'))
        self.assertTrue(self.poll(2.05, report(2), button='GL'))

    def test_disabled_mode_closes_handle_and_cannot_fire(self):
        self.poll(0, report())
        self.assertFalse(self.poll(1, enabled=False))
        self.assertTrue(self.backend.pad.closed)
        self.assertIn('已关闭', self.reader.status)

    def test_disconnect_reconnect_does_not_trigger_from_held_button(self):
        self.poll(0, report())
        self.assertFalse(self.poll(1, OSError('unplugged')))
        self.assertTrue(self.backend.pad.closed)
        self.assertFalse(self.poll(3, report(1)))
        self.assertFalse(self.poll(4, report()))
        self.assertTrue(self.poll(4.05, report(1)))

    def test_no_reports_marks_stale_handle_and_recovers(self):
        self.poll(0, report())
        self.assertFalse(self.poll(4))
        self.assertTrue(self.backend.pad.closed)
        self.assertIn('未收到', self.reader.status)
        self.assertFalse(self.poll(6, report(1)))
        self.assertFalse(self.poll(7, report()))
        self.assertTrue(self.poll(7.05, report(1)))

    def test_missing_or_ambiguous_device_never_selects_arbitrary_pad(self):
        self.backend.devices = []
        self.assertFalse(self.poll(0))
        self.assertIn('未找到', self.reader.status)
        self.backend.devices = [dict(path=b'a', usage_page=1, usage=5, interface_number=0)] * 2
        self.assertFalse(self.poll(2))
        self.assertIn('多个', self.reader.status)


if __name__ == '__main__':
    unittest.main()
