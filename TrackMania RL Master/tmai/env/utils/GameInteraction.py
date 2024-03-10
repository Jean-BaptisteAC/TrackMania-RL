import ctypes
import time
import win32gui
import numpy as np
from enum import Enum
from env.utils.constants import GAME_WINDOW_NAME
import vgamepad as vg

class ArrowInput(Enum):
    UP = 0xC8
    DOWN = 0xD0
    LEFT = 0xCB
    RIGHT = 0xCD
    DEL = 0xD3
    RETURN = 0x1C
    
    def from_discrete_agent_out(vec: np.ndarray) -> list["ArrowInput"]:
        "binary inpuit vector, for each action, 1 if pressed, 0 if not"
        inputs = []
        if vec[0] == 1:
            inputs.append(ArrowInput.LEFT)

        if vec[1] == 1:
            inputs.append(ArrowInput.UP)

        if vec[2] == 1:
            inputs.append(ArrowInput.RIGHT)

        if vec[3] == 1:
            inputs.append(ArrowInput.DOWN)
            
        return inputs


PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


def refocus():
    # refocus on game window decorator, not yet used    
    def wrapper(func):
        def inner(*args, **kwargs):
            hwnd = win32gui.FindWindow(None, args[0].window_name)
            win32gui.SetForegroundWindow(hwnd)
            return func(*args, **kwargs)
        return inner
    
    return wrapper

class KeyboardInputManager:
    def __init__(
        self, input_duration: float = 0.05, window_name: str = GAME_WINDOW_NAME
    ) -> None:
        self.input_duration = input_duration
        self.window_name = window_name

    def press_key(self, key: ArrowInput):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key.value, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(self, key: ArrowInput):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key.value, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def play_inputs(self, inputs: list[ArrowInput]):
        for input_ in inputs:
            if input_ is None:
                continue
            self.press_key(input_)
            time.sleep(self.input_duration)
            self.release_key(input_)

    def play_inputs_no_release(self, inputs: ArrowInput):
        if ArrowInput.UP in inputs:
            self.press_key(ArrowInput.UP)
        else:
            self.release_key(ArrowInput.UP)

        if ArrowInput.DOWN in inputs:
            self.press_key(ArrowInput.DOWN)
        else:
            self.release_key(ArrowInput.DOWN)

        if ArrowInput.LEFT in inputs:
            self.press_key(ArrowInput.LEFT)
        else:
            self.release_key(ArrowInput.LEFT)

        if ArrowInput.RIGHT in inputs:
            self.press_key(ArrowInput.RIGHT)
        else:
            self.release_key(ArrowInput.RIGHT)

