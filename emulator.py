import os.path
import numpy as np
import cv2
from collections import deque
from image_viewer import SimpleImageViewer
from z80wrapper import Z80Wrapper
from ctypes import CDLL, POINTER, c_int, c_byte, c_void_p, cast

########################################################################
class Key:
    Empty = 0
    Shift = 0x001
    Z     = 0x002
    X     = 0x004
    C     = 0x008
    V     = 0x010
    A     = 0x101
    S     = 0x102
    D     = 0x104
    F     = 0x108
    G     = 0x110
    Q     = 0x201
    W     = 0x202
    E     = 0x204
    R     = 0x208
    T     = 0x210
    D1    = 0x301
    D2    = 0x302
    D3    = 0x304
    D4    = 0x308
    D5    = 0x310
    D0    = 0x401
    D9    = 0x402
    D8    = 0x404
    D7    = 0x408
    D6    = 0x410
    P     = 0x501
    O     = 0x502
    I     = 0x504
    U     = 0x508
    Y     = 0x510
    Enter = 0x601
    L     = 0x602
    K     = 0x604
    J     = 0x608
    H     = 0x610
    Space = 0x701
    Sym   = 0x702
    M     = 0x704
    N     = 0x708
    B     = 0x710

########################################################################
class Emulator():
    def __init__(self, path):
        self.z80 = Z80Wrapper(path)
        self.context = self.z80.CreateContext()
        self.path = path
        self.buflen = 352 * 312
        self.screen_buffer = (c_byte * self.buflen)()
        self.screen_buffer_ptr = cast(self.screen_buffer, POINTER(c_byte))

        file_data = self.readfrom(self.path)
        self.rom = (c_byte * len(file_data)).from_buffer(file_data)
        self.rom_ptr = cast(self.rom, POINTER(c_byte))

    def __del__(self):
        self.z80.DestroyContext(self.context)

    def readfrom(self, s):
        if not os.path.exists(s): raise Exception("cannot open file %s" % s)
        fh = open(s, "rb")
        ret = bytearray(fh.read())
        fh.close()
        return ret

    def NextFrame(self): # note that returned screen buffer is overwritten on next call
        r = self.z80.RenderFrame(self.context, self.screen_buffer_ptr, self.buflen)
        if r != 0: raise Exception("render failed")
        return self.screen_buffer

    def KeyUp(self, key):
        self.z80.KeyUp(self.context, key)

    def KeyDown(self, key):
        self.z80.KeyDown(self.context, key)

    def GetByte(self, offset):
        return self.z80.ReadMemory(self.context, offset)

    def SetByte(self, offset, value):
        return self.z80.WriteMemory(self.context, offset, value)

    def Reset(self):
        self.z80.LoadZ80Format(self.context, self.rom_ptr, len(self.rom));
