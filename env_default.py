from frame import Frame
import numpy as np

def nextframe(emu, extra_frames_per_channel):
    ret = emu.NextFrame()
    while extra_frames_per_channel > 0:
        emu.NextFrame()
        extra_frames_per_channel = extra_frames_per_channel - 1
    return ret

def default_render(emu, cut=None, extra_frames_per_channel=0):
    frame1 = Frame.Downsample(nextframe(emu, extra_frames_per_channel), cut)
    frame2 = Frame.Downsample(nextframe(emu, extra_frames_per_channel), cut)
    frame3 = Frame.Downsample(nextframe(emu, extra_frames_per_channel), cut)
    frame4 = Frame.Downsample(nextframe(emu, extra_frames_per_channel), cut)
    return Frame.Join(frame1, frame2, frame3, frame4)

def default_reset(emu, skip=0):
    emu.Reset()
    while skip > 0:
        emu.NextFrame()
        skip = skip - 1

def default_action(emu, action, keys):
    u,d,l,r,f = keys

    emu.KeyUp(u)
    emu.KeyUp(d)
    emu.KeyUp(l)
    emu.KeyUp(r)
    emu.KeyUp(f)
    if   action == 0: 
        pass
    elif action == 1: #up
        emu.KeyDown(u)
    elif action == 2: #down
        emu.KeyDown(d)
    elif action == 3: #left
        emu.KeyDown(l)
    elif action == 4: #right
        emu.KeyDown(r)
    elif action == 5: #upleft
        emu.KeyDown(u)
        emu.KeyDown(l)
    elif action == 6: #upright
        emu.KeyDown(u)
        emu.KeyDown(r)
    elif action == 7: #downleft
        emu.KeyDown(d)
        emu.KeyDown(l)
    elif action == 8: #downright
        emu.KeyDown(d)
        emu.KeyDown(r)
    elif action == 9: #fire
        emu.KeyDown(f)
    elif action == 10: #upfire
        emu.KeyDown(f)
        emu.KeyDown(u)
    elif action == 11: #downfire
        emu.KeyDown(f)
        emu.KeyDown(d)
    elif action == 12: #leftfire
        emu.KeyDown(f)
        emu.KeyDown(l)
    elif action == 13: #rightfire
        emu.KeyDown(f)
        emu.KeyDown(r)
    elif action == 14: #upleftfire
        emu.KeyDown(f)
        emu.KeyDown(u)
        emu.KeyDown(l)
    elif action == 15: #uprightfire
        emu.KeyDown(f)
        emu.KeyDown(u)
        emu.KeyDown(r)
    elif action == 16: #downleftfire
        emu.KeyDown(f)
        emu.KeyDown(d)
        emu.KeyDown(l)
    elif action == 17: #downrightfire
        emu.KeyDown(f)
        emu.KeyDown(d)
        emu.KeyDown(r)
    else: raise Exception("Error")
