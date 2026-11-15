"""SDL backend contracts without real devices, UI windows, or user settings."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from controller_sdl import SDLReader


def pad(instance=1, *, path=b"device-path-a", guid=b"a" * 16, name=b"Test pad"):
    return SimpleNamespace(instance=instance, path=path, guid=guid, name=name,
                           buttons=(False, True), hats=(0, 3), axes=(-32768, 0, 32767),
                           attached=True)


class FakeSDL:
    SDL_INIT_JOYSTICK = 0x200
    SDL_IGNORE = 0

    def __init__(self):
        self.devices = [pad()]
        self.opened = []
        self.closed = []
        self.init_flags = []
        self.quit_flags = []
        self.hints = []
        self.event_state = None
        self.init_result = 0
        self.fail_open = False
        self.updates = 0

    def SDL_SetHint(self, name, value):
        self.hints.append((name, value))
        return 1

    def SDL_InitSubSystem(self, flags):
        self.init_flags.append(flags)
        return self.init_result

    def SDL_QuitSubSystem(self, flags):
        self.quit_flags.append(flags)

    def SDL_JoystickEventState(self, state):
        self.event_state = state

    def SDL_JoystickUpdate(self):
        self.updates += 1

    def SDL_NumJoysticks(self):
        return len(self.devices)

    def SDL_JoystickGetDeviceGUID(self, index):
        return SimpleNamespace(data=self.devices[index].guid)

    def SDL_JoystickPathForIndex(self, index):
        return self.devices[index].path

    def SDL_JoystickGetDeviceInstanceID(self, index):
        return self.devices[index].instance

    def SDL_JoystickNameForIndex(self, index):
        return self.devices[index].name

    def SDL_JoystickOpen(self, index):
        self.opened.append(index)
        return None if self.fail_open else self.devices[index]

    def SDL_JoystickClose(self, handle):
        self.closed.append(handle)

    def SDL_JoystickGetAttached(self, handle):
        return handle.attached and any(d is handle for d in self.devices)

    def SDL_JoystickInstanceID(self, handle):
        return handle.instance

    def SDL_JoystickNumButtons(self, handle):
        return len(handle.buttons)

    def SDL_JoystickNumHats(self, handle):
        return len(handle.hats)

    def SDL_JoystickNumAxes(self, handle):
        return len(handle.axes)

    def SDL_JoystickGetButton(self, handle, index):
        return handle.buttons[index]

    def SDL_JoystickGetHat(self, handle, index):
        return handle.hats[index]

    def SDL_JoystickGetAxis(self, handle, index):
        return handle.axes[index]


class SDLReaderTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeSDL()
        self.now = 0.0
        self.reader = SDLReader(self.api, clock=lambda: self.now)
        self.addCleanup(self.reader.close)

    def device_id(self, index=0):
        return self.reader.list_devices()[index]["id"]

    def test_initialization_uses_joystick_only_with_background_input(self):
        self.assertEqual(self.api.init_flags, [self.api.SDL_INIT_JOYSTICK])
        self.assertEqual(self.api.hints, [(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")])
        self.assertEqual(self.api.event_state, self.api.SDL_IGNORE)

    def test_enumeration_opens_no_devices_and_hides_raw_paths(self):
        devices = self.reader.list_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "Test pad")
        self.assertEqual(devices[0]["backend"], "sdl")
        self.assertNotIn("device-path", devices[0]["id"])
        self.assertEqual(self.api.opened, [])

    def test_reads_all_control_types_without_events_or_video(self):
        state = self.reader.read(self.device_id())
        self.assertEqual(state, {"buttons": (False, True), "hats": (0, 3),
                                 "axes": (-1.0, 0.0, 1.0), "connection": 1})
        self.assertEqual(self.api.opened, [0])
        self.reader.read(self.device_id())
        self.assertEqual(self.api.opened, [0])
        self.assertGreater(self.api.updates, 0)

    def test_different_paths_distinguish_same_model_and_only_selected_opens(self):
        self.api.devices.append(pad(2, path=b"device-path-b"))
        devices = self.reader.list_devices()
        self.assertNotEqual(devices[0]["id"], devices[1]["id"])
        self.assertEqual(self.reader.read(devices[1]["id"])["connection"], 2)
        self.assertEqual(self.api.opened, [1])
        self.reader.read(devices[0]["id"])
        self.assertEqual(self.api.opened, [1, 0])
        self.assertEqual(len(self.api.closed), 1)

    def test_same_identity_is_ambiguous_instead_of_selecting_arbitrarily(self):
        self.api.devices = [pad(1, path=None), pad(2, path=None)]
        devices = self.reader.list_devices()
        self.assertEqual(len(devices), 1)
        self.assertTrue(devices[0]["ambiguous"])
        self.assertIsNone(self.reader.read(devices[0]["id"]))
        self.assertEqual(self.api.opened, [])
        self.assertIn("同标识", self.reader.status)

    def test_reconnect_keeps_binding_id_and_changes_connection_token(self):
        identity = self.device_id()
        first = self.reader.read(identity)
        self.api.devices.clear()
        self.assertIsNone(self.reader.read(identity))
        self.assertEqual(len(self.api.closed), 1)
        self.api.devices.append(pad(8))
        self.now = 1.0
        second = self.reader.read(identity)
        self.assertEqual(self.device_id(), identity)
        self.assertEqual(first["connection"], 1)
        self.assertEqual(second["connection"], 8)
        self.assertEqual(second["buttons"], (False, True))

    def test_index_reorder_does_not_rebind_an_open_handle(self):
        self.api.devices.append(pad(2, path=b"other"))
        identity = self.device_id()
        self.reader.read(identity)
        self.api.devices.reverse()
        self.now = 1.0
        self.assertEqual(self.reader.read(identity)["connection"], 1)
        self.assertEqual(self.api.opened, [0])

    def test_disappearing_index_cannot_open_a_different_device(self):
        identity = self.device_id()
        with patch.object(self.api, "SDL_JoystickGetDeviceInstanceID", return_value=99):
            self.assertIsNone(self.reader.read(identity))
        self.assertEqual(self.api.opened, [])

    def test_failed_open_recovers_on_next_read(self):
        identity = self.device_id()
        self.api.fail_open = True
        self.assertIsNone(self.reader.read(identity))
        self.api.fail_open = False
        self.assertEqual(self.reader.read(identity)["connection"], 1)

    def test_read_failure_closes_handle_and_recovers(self):
        identity = self.device_id()
        with patch.object(self.api, "SDL_JoystickNumButtons", return_value=-1):
            self.assertIsNone(self.reader.read(identity))
        self.assertEqual(len(self.api.closed), 1)
        self.assertIsNotNone(self.reader.read(identity))

    def test_close_is_idempotent_and_scoped_to_joystick(self):
        identity = self.device_id()
        self.reader.read(identity)
        self.reader.close()
        self.reader.close()
        self.assertEqual(self.api.quit_flags, [self.api.SDL_INIT_JOYSTICK])
        self.assertEqual(len(self.api.closed), 1)
        self.assertIsNone(self.reader.read(identity))
        self.assertEqual(self.reader.list_devices(), [])

    def test_release_closes_only_selection_and_allows_reopening(self):
        identity = self.device_id()
        self.reader.read(identity)
        self.reader.release()
        self.reader.release()
        self.assertEqual(len(self.api.closed), 1)
        self.assertEqual(self.api.quit_flags, [])
        self.assertEqual(self.device_id(), identity)
        self.assertEqual(self.reader.read(identity)["connection"], 1)
        self.assertEqual(self.api.opened, [0, 0])

    def test_import_and_initialization_failures_leave_adapter_available(self):
        with patch("controller_sdl.importlib.import_module", side_effect=ImportError("test")):
            missing = SDLReader()
        self.assertEqual(missing.list_devices(), [])
        self.assertIsNone(missing.read("anything"))
        self.assertIn("加载失败", missing.status)
        missing.close()
        api = FakeSDL()
        api.init_result = -1
        failed = SDLReader(api)
        self.assertEqual(failed.list_devices(), [])
        self.assertIsNone(failed.read("anything"))
        failed.close()
        self.assertEqual(api.quit_flags, [])

    def test_cleanup_errors_stay_isolated_from_other_controller_backends(self):
        identity = self.device_id()
        self.reader.read(identity)
        with patch.object(self.api, "SDL_JoystickClose", side_effect=OSError("test")), \
                patch.object(self.api, "SDL_QuitSubSystem", side_effect=OSError("test")):
            self.reader.close()
        self.assertIsNone(self.reader.read(identity))
        self.assertEqual(self.reader.list_devices(), [])


if __name__ == "__main__":
    unittest.main()
