import os.path
import numpy as np
import cv2
from collections import deque
from image_viewer import SimpleImageViewer
from ctypes import CDLL, POINTER, c_int, c_byte, c_void_p, cast
from z80wrapper import Z80Wrapper

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
colorbits = {
      0:0x00000000 ,
      1:0x00D70000 ,
      2:0x000000D7 ,
      3:0x00D700D7 ,
      4:0x0000D700 ,
      5:0x00D7D700 ,
      6:0x0000D7D7 ,
      7:0x00D7D7D7 ,
      8:0x00000000 ,
      9:0x00FF0000 ,
     10:0x000000FF ,
     11:0x00FF00FF ,
     12:0x0000FF00 ,
     13:0x00FFFF00 ,
     14:0x0000FFFF ,
     15:0x00FFFFFF }

palette = np.vectorize(colorbits.get, otypes=[np.uint32])

#screen = bytearray(312 * 352 * 3)
step = 0
while True:
  if step % 2 == 0:
    emu.KeyDown(0x701) 
  else:
    emu.KeyUp(0x701)
  frame = emu.NextFrame()
  screen = np.frombuffer(palette(frame).tobytes(), dtype=np.uint8)
  #for i in range(len(frame)):
  #  b = frame[i]
  #  screen[i*3+0] = colors[b][0]
  #  screen[i*3+1] = colors[b][1]
  #  screen[i*3+2] = colors[b][2]

  viewer.imshow(np.reshape(screen, (312,352,4)))
  step += 1

