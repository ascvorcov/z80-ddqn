class Renderer
  def render(self):
    pass


class NormalRender(Renderer):
  def __init__(self, render_mode = 0):
    self.render_mode = render_mode

  def render(self):
    pass


class GifRender(Renderer):
  def render(self):
    pass


class HeatmapRender(Renderer):
  def render(self):
    pass
