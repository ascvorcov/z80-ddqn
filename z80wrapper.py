from ctypes import CDLL, POINTER, c_int, c_byte, c_void_p, cast

class Z80Wrapper:
    def __init__(self, path):
        dll = CDLL('z80native.dll')
        self.CreateContext = dll.CreateContext
        self.CreateContext.restype = c_void_p

        self.DestroyContext = dll.DestroyContext
        self.DestroyContext.argtypes = [c_void_p]

        self.LoadZ80Format = dll.LoadZ80Format
        self.LoadZ80Format.argtypes = [c_void_p, POINTER(c_byte), c_int]

        self.KeyDown = dll.KeyDown
        self.KeyDown.argtypes = [c_void_p, c_int]

        self.KeyUp = dll.KeyUp
        self.KeyUp.argtypes = [c_void_p, c_int]

        self.RenderFrame = dll.RenderFrame
        self.RenderFrame.argtypes = [c_void_p, POINTER(c_byte), c_int]
        self.RenderFrame.restype = c_int

        self.ReadMemory = dll.ReadMemory
        self.ReadMemory.argtypes = [c_void_p, c_int]
        self.ReadMemory.restype = c_int

        self.WriteMemory = dll.WriteMemory
        self.WriteMemory.argtypes = [c_void_p, c_int, c_byte]
        self.WriteMemory.restype = c_int
