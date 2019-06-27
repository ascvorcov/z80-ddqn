import os.path
import numpy as np
import ctypes
import cv2
from collections import deque
from image_viewer import SimpleImageViewer

########################################################################
class Emulator():
    def __init__(self, path):
        self.z80 = ctypes.cdll.LoadLibrary('z80native.dll')
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
        if not os.path.exists(s): raise Exception('cannot open file %s' % s)
        fh = open(s, 'rb')
        ret = bytearray(fh.read())
        fh.close()
        return ret

    def NextFrame(self):
        r = self.z80.RenderFrame(self.context, self.screen_buffer_ptr, self.buflen)
        if r != 0: raise Exception('render failed')
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
emu = Emulator('./roms/riverraid.z80')
emu.Reset()
viewer = SimpleImageViewer() #expects HWC image
colors = [
      [0,0,0],
      [0,0,0xD7],
      [0xD7,0,0],
      [0xD7,0,0xD7],
      [0,0xD7,0],
      [0,0xD7,0xD7],
      [0xD7,0xD7,0],
      [0xD7,0xD7,0xD7],
      [0,0,0],
      [0,0,0xFF],
      [0xFF,0,0],
      [0xFF,0,0xFF],
      [0,0xFF,0],
      [0,0xFF,0xFF],
      [0xFF,0xFF,0],
      [0xFF,0xFF,0xFF]]

screen = bytearray(312 * 352 * 3)
step = 0
while True:
  if step % 2 == 0:
    emu.KeyDown(0x701) 
  else:
    emu.KeyUp(0x701)
  frame = emu.NextFrame()
  for i in range(len(frame)):
    b = frame[i]
    screen[i*3+0] = colors[b][0]
    screen[i*3+1] = colors[b][1]
    screen[i*3+2] = colors[b][2]

  viewer.imshow(np.reshape(screen, (312,352,3)))
  step += 1
