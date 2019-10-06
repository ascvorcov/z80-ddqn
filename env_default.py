from frame import Frame

import numpy as np
import imageio
import os

_gif_writer_frame = None
_gif_writer_state = None

_colorbits = {
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

_palette = np.vectorize(_colorbits.get, otypes=[np.uint32])

def nextframe(emu, extra_frames_per_channel):
    ret = bytearray(emu.NextFrame())
    while extra_frames_per_channel > 0:
        emu.NextFrame()
        extra_frames_per_channel = extra_frames_per_channel - 1
    return ret

def default_next_frame(emu, cut=None, extra_frames_per_channel=0, filter_image=False):
    raw1 = nextframe(emu, extra_frames_per_channel)
    raw2 = nextframe(emu, extra_frames_per_channel)
    raw3 = nextframe(emu, extra_frames_per_channel)
    raw4 = nextframe(emu, extra_frames_per_channel)

    frame1 = Frame.Downsample(raw1, cut, filter_image)
    frame2 = Frame.Downsample(raw2, cut, filter_image)
    frame3 = Frame.Downsample(raw3, cut, filter_image)
    frame4 = Frame.Downsample(raw4, cut, filter_image)

    next_state = Frame.Join(frame1, frame2, frame3, frame4)

    return (raw1, next_state)

def default_render(viewer, render_mode, state, frame):
    global _gif_writer_frame
    global _gif_writer_state

    if state != None and (render_mode == 2 or render_mode == 3):
        screen = np.asarray(state)
        arr = np.swapaxes(screen, 0, 2).astype(np.uint8)
        if render_mode == 3:
            _gif_writer_state.append_data(arr)
        if render_mode == 2:
            viewer.imshow(arr)

    if frame != None and (render_mode == 1 or render_mode == 3):
        screen = np.frombuffer(_palette(frame).tobytes(), dtype=np.uint8)
        arr = np.reshape(screen, (312,352,4))
        if render_mode == 3:
            _gif_writer_frame.append_data(arr)
        viewer.imshow(arr)

def default_reset(emu, render_mode, skip=0):
    global _gif_writer_frame
    global _gif_writer_state

    if _gif_writer_frame != None:
        _gif_writer_frame.close()
        move_file("frame.gif", "./gifs")

    if _gif_writer_state != None:
        _gif_writer_state.close()
        move_file("state.gif", "./gifs")

    if render_mode == 3:
        _gif_writer_frame = imageio.get_writer(uri="./frame.gif", mode="I")
        _gif_writer_state = imageio.get_writer(uri="./state.gif", mode="I")

    emu.Reset()
    while skip > 0:
        emu.NextFrame()
        skip = skip - 1

def move_file(fname, folder):
    if not os.path.isfile(fname): return
    if not os.path.isdir(folder):
        os.makedirs(folder)
    idx = 1
    while os.path.isfile(folder + "/" + str(idx) + fname):
        idx = idx + 1
    os.rename(fname, folder + "/" + str(idx) + fname)


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
