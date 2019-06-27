import os.path
import numpy as np
import ctypes
import cv2
from collections import deque
from image_viewer import SimpleImageViewer

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
class Frame():
    def __init__(self, arr):
        self.arr = arr

    def __array__(self, dtype=None):
        out = np.frombuffer(self.arr, dtype=np.uint8)
        out = np.reshape(out, (4, 84, 84)).astype(np.float32) / 252.0

        if dtype is not None:
            out = out.astype(dtype)
        return out

    @staticmethod
    @profile
    def Downsample(frame):
        # expected frame size is 352x312 (including border).
        if len(frame) != 352*312: raise Exception('Unexpected size')

        # grayscale conversion lookup table from color index
        lookup = [0,6,16,22,31,37,47,53,0,7,19,26,37,44,56,63]
        LeftBorder = 48
        TopBorder = 55
        LineSize = 352
        destination = bytearray(84*84)
        view = memoryview(destination)
        scanline = view[:] # todo: pack two 16-color bytes into one
        destinationIndex = 0
        for y in range(84*2):
            if (y % 2) == 0 and y > 0:
                destinationIndex = destinationIndex + 84
                scanline = view[destinationIndex:]
            
            offset = ((y + TopBorder) * LineSize) + LeftBorder + 44
            for x in range(84):# each cell is an average of 4 adjacent cells (2x2), max value is 252
                scanline[x] += lookup[frame[offset + 0]] + lookup[frame[offset + 1]]
                offset += 2
                    
        return destination

    def Join(*argv):
        result = bytearray(0)
        for arr in argv:
            result += arr
        return Frame(result)

########################################################################
class MainGymWrapper():
    def __init__(self, name):
        self.name = name
        if name == 'RiverRaid':
            self.env = RiverRaidEnv()
        elif name == 'R-Type':
            self.env = RTypeEnv()
        elif name == 'Exolon':
            self.env = ExolonEnv()

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)

########################################################################
class ExolonEnv():
    def __init__(self):
        pass

########################################################################
class RTypeEnv():
    def __init__(self):
        pass

########################################################################
class RiverRaidEnv():
    def __init__(self):
        self.action_space = 18
        self.lives = 3
        self.score = 0
        self.lookup = {0x4A: 0,0x08: 1,0x02: 2,0x0C: 3,0x28: 4,0x04: 7,0x3C: 8,0x42: 9}
        self.emu = Emulator('./roms/riverraid.z80')
        self.latestFrame = None
        self.viewer = None

    def reset(self):
        self.lives = 3
        self.score = 0
        emu = self.emu
        emu.Reset()
        frame1 = Frame.Downsample(emu.NextFrame())
        frame2 = Frame.Downsample(emu.NextFrame())
        frame3 = Frame.Downsample(emu.NextFrame())
        frame4 = Frame.Downsample(emu.NextFrame())
        next_state = self.latestFrame = Frame.Join(frame1, frame2, frame3, frame4)
        return next_state

    def render(self):
        if self.latestFrame == None: return
        if self.viewer == None:
            self.viewer = SimpleImageViewer()
        arr = np.moveaxis(np.asarray(self.latestFrame), 0, 2)
        self.viewer.imshow((arr * 255.0).astype(np.uint8))

    def step(self, action):
        emu = self.emu
        emu.KeyUp(Key.O);
        emu.KeyUp(Key.P);
        emu.KeyUp(Key.W);
        emu.KeyUp(Key.D2);
        emu.KeyUp(Key.Space);
        if   action == 0: 
            pass
        elif action == 1: 
            emu.KeyDown(Key.D2)
        elif action == 2:
            emu.KeyDown(Key.W)
        elif action == 3:
            emu.KeyDown(Key.O)
        elif action == 4:
            emu.KeyDown(Key.P)
        elif action == 5:
            emu.KeyDown(Key.D2)
            emu.KeyDown(Key.O)
        elif action == 6:
            emu.KeyDown(Key.D2)
            emu.KeyDown(Key.P)
        elif action == 7:
            emu.KeyDown(Key.W)
            emu.KeyDown(Key.O)
        elif action == 8:
            emu.KeyDown(Key.W)
            emu.KeyDown(Key.P)
        elif action == 9:
            emu.KeyDown(Key.Space)
        elif action == 10:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.D2)
        elif action == 11:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.W)
        elif action == 12:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.O)
        elif action == 13:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.P)
        elif action == 14:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.D2)
            emu.KeyDown(Key.O)
        elif action == 15:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.D2)
            emu.KeyDown(Key.P)
        elif action == 16:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.W)
            emu.KeyDown(Key.O)
        elif action == 17:
            emu.KeyDown(Key.Space)
            emu.KeyDown(Key.W)
            emu.KeyDown(Key.P)
        else: raise Exception("Error")

        frame1 = Frame.Downsample(emu.NextFrame())
        frame2 = Frame.Downsample(emu.NextFrame())
        frame3 = Frame.Downsample(emu.NextFrame())
        frame4 = Frame.Downsample(emu.NextFrame())

        next_state = Frame.Join(frame1, frame2, frame3, frame4)
        reward = self.UpdateReward();
        terminal = self.UpdateLivesAndRewindIfPlayerDied();
        self.latestFrame = next_state
        return (next_state, reward, terminal)

    def UpdateLivesAndRewindIfPlayerDied(self):
        emu = self.emu
        newLives = emu.GetByte(0x923B)
        oldLives = self.lives;
        self.lives = newLives;

        speed = emu.GetByte(0x5F64) # 0 - stopped/crashed, 1,2,4 - slow/normal/fast
        if speed == 0: # player died, rewind death animation
            if self.lives == 0: # terminal state
                return True;

            while emu.GetByte(0x5F69) != 4:
                emu.NextFrame();

        return False;

    def UpdateReward(self):
        newScore = self.ReadScore()
        if newScore == 0: return 0
        reward = newScore - self.score
        self.score = newScore
        return reward

    def ReadScore(self):
        # read score directly from screen
        emu = self.emu
        b1 = emu.GetByte(0x53E5)
        b2 = emu.GetByte(0x53E6)
        b3 = emu.GetByte(0x53E7)
        b4 = emu.GetByte(0x53E8)
        b5 = emu.GetByte(0x53E9)
        b6 = emu.GetByte(0x53EA)
        b7 = emu.GetByte(0x53EB)
        
        h1 = emu.GetByte(0x54E5)
        h2 = emu.GetByte(0x54E6)
        h3 = emu.GetByte(0x54E7)
        h4 = emu.GetByte(0x54E8)
        h5 = emu.GetByte(0x54E9)
        h6 = emu.GetByte(0x54EA)
        h7 = emu.GetByte(0x54EB)
        
        d1 = self.GetDigit(b1,h1);
        d2 = self.GetDigit(b2,h2);
        d3 = self.GetDigit(b3,h3);
        d4 = self.GetDigit(b4,h4);
        d5 = self.GetDigit(b5,h5);
        d6 = self.GetDigit(b6,h6);
        d7 = self.GetDigit(b7,h7);
        
        return d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7;

    def GetDigit(self, mem1, mem2):
        if mem1 in self.lookup:
            return self.lookup[mem1]
        elif mem2 == 2 and mem1 == 0x7C:
            return 5
        elif mem2 == 0x42 and mem1 == 0x7C:
            return 6
        return 0
