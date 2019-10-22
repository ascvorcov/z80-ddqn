import os.path
import numpy as np
import ctypes
import cv2
from collections import deque
from image_viewer import SimpleImageViewer
from z80wrapper import Z80Wrapper

########################################################################
class Emulator():
    def __init__(self, path):
        self.z80 = Z80Wrapper(path)
        self.context = self.z80.CreateContext()
        self.path = path
        self.buflen = 352 * 312
        self.screen_buffer = (ctypes.c_byte * self.buflen)()
        self.screen_buffer_ptr = ctypes.cast(self.screen_buffer, ctypes.POINTER(ctypes.c_byte))

        file_data = self.readfrom(self.path)
        self.rom = (ctypes.c_byte * len(file_data)).from_buffer(file_data)
        self.rom_ptr = ctypes.cast(self.rom, ctypes.POINTER(ctypes.c_byte))

    def __del__(self):
        self.z80.DestroyContext(self.context)

    def readfrom(self, s):
        if not os.path.exists(s): raise Exception("cannot open file %s" % s)
        fh = open(s, "rb")
        ret = bytearray(fh.read())
        fh.close()
        return ret

    def NextFrame(self):
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

########################################################################
emu = Emulator("./roms/xecutor.z80")
emu.Reset()
viewer = SimpleImageViewer() #expects HWC image

screen = bytearray(312 * 352 * 3)
step = 0

lookup = {0:0,1:6,2:16,3:22,4:31,5:37,6:47,7:53,8:0,9:7,10:19,11:26,12:37,13:44,14:56,15:63}
palette = np.vectorize(lookup.get, otypes=[np.uint8])

while True:
  if step % 2 == 0:
    emu.KeyDown(0x701) 
  else:
    emu.KeyUp(0x701)
  frame = np.frombuffer(emu.NextFrame(), dtype=np.uint8).reshape(312,352)
  img = frame[64:-56,80:-80] # cut part of img 192x192
  img = palette(img).reshape(192,192,1) # remap palette and reshape
  img = cv2.resize(img*16, (96, 96), interpolation=cv2.INTER_AREA).reshape(96,96,1) # resize with interpolation and reshape
  viewer.imshow(img)
  step += 1

