import copy
import unittest
from unittest.mock import Mock, patch

from controller_binding import binding_label, legacy_binding, normalize_binding
from controller_input import BindingLearner, BindingTrigger


DEVICE = {"id": "sdl:device-one", "backend": "sdl", "name": "Example gamepad"}


def binding(kind="button", index=0, direction=0, **extra):
    return dict(backend="sdl", device=DEVICE["id"], name=DEVICE["name"],
                kind=kind, index=index, direction=direction, **extra)


def state(buttons=(), hats=(), axes=(), connection=1):
    return dict(buttons=buttons, hats=hats, axes=axes, connection=connection)


class ControllerInputTests(unittest.TestCase):
    def test_generic_buttons_and_axis_bindings_survive_normalization(self):
        for value in (binding(index=45), binding("hat", 1, 8), binding("axis", 4, 1, rest=-1), legacy_binding("GL")):
            self.assertEqual(normalize_binding(value), value)
            self.assertNotEqual(binding_label(value), "待录入翻译按键")

    def test_invalid_bindings_remain_unbound(self):
        for value in (None, [], {}, binding(index=True), binding(index=-1), binding(direction=2),
                      binding("hat", 0, 3), binding("axis", 0, 1, rest=float("nan")),
                      binding("axis", 0, 1, rest=2), dict(binding(), device="switch2:wrong")):
            self.assertEqual(normalize_binding(value), {})

    def test_any_button_hold_repeat_and_reconnect_require_a_release(self):
        trigger = BindingTrigger()
        chosen = binding(index=31)
        held = state([False] * 31 + [True])
        idle = state([False] * 32)
        self.assertFalse(trigger.update(chosen, held, 0))
        self.assertFalse(trigger.update(chosen, idle, 1))
        self.assertTrue(trigger.update(chosen, held, 1.1))
        self.assertFalse(trigger.update(chosen, held, 5))
        self.assertFalse(trigger.update(chosen, None, 6))
        self.assertFalse(trigger.update(chosen, dict(held, connection=2), 7))
        self.assertFalse(trigger.update(chosen, dict(idle, connection=2), 8))
        self.assertTrue(trigger.update(chosen, dict(held, connection=2), 8.1))

    def test_hat_uses_selected_direction_and_axis_uses_rest_and_hysteresis(self):
        trigger = BindingTrigger()
        chosen = binding("hat", 0, 1)
        self.assertFalse(trigger.update(chosen, state(hats=[0]), 0))
        self.assertFalse(trigger.update(chosen, state(hats=[2]), 0.1))
        self.assertTrue(trigger.update(chosen, state(hats=[3]), 0.2))
        chosen = binding("axis", 0, 1, rest=-1.0)
        self.assertFalse(trigger.update(chosen, state(axes=[-1]), 1))
        self.assertFalse(trigger.update(chosen, state(axes=[-0.6]), 1.1))
        self.assertTrue(trigger.update(chosen, state(axes=[0.0]), 1.2))
        self.assertFalse(trigger.update(chosen, state(axes=[-0.6]), 2))
        self.assertFalse(trigger.update(chosen, state(axes=[0.0]), 2.1))
        self.assertFalse(trigger.update(chosen, state(axes=[-1]), 3))
        self.assertTrue(trigger.update(chosen, state(axes=[1]), 3.1))

    def test_learning_needs_initial_release_and_a_complete_new_press(self):
        learner = BindingLearner(DEVICE, False, 0)
        self.assertIsNone(learner.update(state([True, False]), 0))
        self.assertIsNone(learner.update(state([False, False]), 1))
        self.assertIsNone(learner.update(state([False, False]), 1.4))
        self.assertIsNone(learner.update(state([False, True]), 2))
        self.assertIsNone(learner.update(state([False, True]), 3))
        self.assertEqual(learner.update(state([False, False]), 3.1), binding(index=1))

    def test_analog_motion_cannot_block_or_capture_button_only_learning(self):
        learner = BindingLearner(DEVICE, False, 0)
        learner.update(state([False], axes=[-1]), 0)
        learner.update(state([False], axes=[1]), 0.4)
        learner.update(state([True], axes=[0]), 0.5)
        self.assertEqual(learner.update(state([False], axes=[0.8]), 0.6), binding())

    def test_learning_trigger_records_its_rest_value_and_direction(self):
        learner = BindingLearner(DEVICE, True, 0)
        learner.update(state(axes=[0, -1]), 0)
        learner.update(state(axes=[0, -1]), 0.4)
        learner.update(state(axes=[0, 1]), 0.5)
        found = learner.update(state(axes=[0, -1]), 0.6)
        self.assertEqual(found, binding("axis", 1, 1, rest=-1.0))

    def test_learning_hat_and_switch2_rear_buttons_share_the_same_flow(self):
        for device, idle, pressed, expected in (
            (DEVICE, state(hats=[0]), state(hats=[4]), binding("hat", 0, 4)),
            ({"id": "switch2:auto", "backend": "switch2", "name": "Switch 2 Pro USB"},
             state([False, False]), state([False, True]), legacy_binding("GL")),
        ):
            learner = BindingLearner(device, False, 0)
            learner.update(idle, 0)
            learner.update(idle, 0.4)
            learner.update(pressed, 0.5)
            self.assertEqual(learner.update(idle, 0.6), expected)

    def test_multiple_buttons_and_disconnects_rearm_instead_of_committing(self):
        learner = BindingLearner(DEVICE, False, 0)
        learner.update(state([False, False]), 0)
        learner.update(state([False, False]), 0.4)
        self.assertIsNone(learner.update(state([True, True]), 0.5))
        learner.update(state([False, False]), 1)
        learner.update(state([False, False]), 1.4)
        learner.update(state([True, False]), 1.5)
        self.assertIsNone(learner.update(None, 1.6))
        self.assertIsNone(learner.update(state([False, False], connection=2), 1.7))
        self.assertIsNone(learner.update(state([False, False], connection=2), 2.1))
        learner.update(state([False, True], connection=2), 2.2)
        self.assertEqual(learner.update(state([False, False], connection=2), 2.3), binding(index=1))

    def test_learning_times_out_while_device_is_absent(self):
        learner = BindingLearner(DEVICE, False, 0)
        learner.update(None, 21)
        self.assertTrue(learner.expired)

    def test_second_key_during_a_candidate_hold_rejects_the_chord(self):
        learner = BindingLearner(DEVICE, False, 0)
        learner.update(state([False, False]), 0)
        learner.update(state([False, False]), 0.4)
        learner.update(state([True, False]), 0.5)
        self.assertIsNone(learner.update(state([True, True]), 0.6))
        self.assertIsNone(learner.update(state([False, False]), 0.7))


class ControllerServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from controller import ControllerService
        self.translate = Mock()
        with patch("controller.threading.Thread"):
            self.service = ControllerService(self.app, self.translate, binding=binding())
        self.addCleanup(self.service.close)

    def test_learning_and_reconfiguration_invalidate_queued_translation(self):
        self.service._deliver_press(0)
        self.translate.assert_called_once()
        self.translate.reset_mock()
        self.service.start_learning(DEVICE["id"])
        self.service._deliver_press(0)
        self.service._deliver_press(self.service._version)
        self.translate.assert_not_called()
        self.service.cancel_learning()
        self.service.configure(False, binding=binding())
        self.service._deliver_press(self.service._version)
        self.translate.assert_not_called()

    def test_cancelled_learned_signal_cannot_modify_settings(self):
        learned = Mock()
        self.service.binding_learned.connect(learned)
        self.service.start_learning(DEVICE["id"])
        old_version = self.service._version
        self.service.cancel_learning()
        self.service._deliver_binding(old_version, binding(index=1))
        learned.assert_not_called()
        self.service.start_learning(DEVICE["id"])
        before = copy.deepcopy(self.service._settings)
        self.service._deliver_binding(self.service._version, binding(index=1))
        learned.assert_called_once_with(binding(index=1))
        self.assertEqual(self.service._settings, before)
        self.assertFalse(self.service.learning)

    def test_close_suppresses_queued_inputs(self):
        self.service.close()
        self.service._deliver_press(self.service._version)
        self.translate.assert_not_called()
