import imageio
import os
import numpy as np

from image_viewer import SimpleImageViewer
from heatmap import features_heatmap

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


####################################
class Renderer:
  def render(self, state, frame):
    pass
  def reset(self):
    pass

####################################
class CompositeRender(Renderer):
  def __init__(self, inner):
    self.inner = inner
  def render(self, state, frame):
    for o in self.inner: o.render(state,frame)
  def reset(self):
    for o in self.inner: o.reset()


####################################
class FrameRender(Renderer):
  def __init__(self):
    self.viewer = SimpleImageViewer()

  def render(self, state, frame):
    arr = self.prepare(state, frame)
    if arr is not None: self.viewer.imshow(arr)

  def prepare(self, state, frame):
    if frame is None: return None
    screen = np.frombuffer(_palette(frame).tobytes(), dtype=np.uint8)
    arr = np.reshape(screen, (312,352,4))
    return arr


####################################
class StateRender(Renderer):
  def __init__(self):
    self.viewer = SimpleImageViewer()

  def render(self, state, frame):
    arr = self.prepare(state, frame)
    if arr is not None: self.viewer.imshow(arr)

  def prepare(self, state, frame):
    if state is None: return None
    screen = np.asarray(state)
    arr = np.swapaxes(screen, 0, 2).astype(np.uint8)
    return arr


####################################
class GifRender(Renderer):
  def __init__(self, inner, fname = "./frame.gif"):
    self.writer = None
    self.inner = inner
    self.fname = fname

  def render(self,state,frame):
    arr = self.inner.prepare(state,frame)
    if arr is not None: self.writer.append_data(arr)

  def reset(self):
    if self.writer is not None:
      self.writer.close()
      self.move_file(self.fname, "./gifs")
    self.writer = imageio.get_writer(uri=self.fname, mode="I")

  def move_file(self, fname, folder):
    if not os.path.isfile(fname): return
    if not os.path.isdir(folder):
      os.makedirs(folder)
    idx = 1
    while os.path.isfile(os.path.join(folder, str(idx) + os.path.basename(fname))):
      idx = idx + 1
    os.rename(fname, os.path.join(folder, str(idx) + os.path.basename(fname)))


####################################
class HeatmapRender(Renderer):
  def __init__(self, model, layer_name="conv2d_3", stack_by=8):
    self.model = model
    self.layer_name = layer_name
    self.stack_by = stack_by
    self.viewer = SimpleImageViewer()

  def render(self, state, frame):
    data = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
    hm = features_heatmap(self.model, data, self.layer_name, self.stack_by)
    self.viewer.imshow(hm)

def get_renderer(render_mode, model):
  if   render_mode == 0: return Renderer()
  elif render_mode == 1: return FrameRender()
  elif render_mode == 2: return StateRender()
  elif render_mode == 3: return CompositeRender([FrameRender(),GifRender(FrameRender()),GifRender(StateRender(),"./state.gif")])
  elif render_mode == 4: return HeatmapRender(model)

