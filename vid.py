import imageio
import numpy as np

# All images must be of the same size
image1 = np.stack([imageio.imread('imageio:camera.png')] * 3, 2)
image2 = imageio.imread('imageio:astronaut.png')
image3 = imageio.imread('imageio:immunohistochemistry.png')

w = imageio.get_writer('my_video.mp4', format='FFMPEG', mode='I', fps=1,
                       codec='h264_vaapi',
                       output_params=['-vaapi_device',
                                      '/dev/dri/renderD128',
                                      '-vf',
                                      'format=gray|nv12,hwupload'],
                       pixelformat='vaapi_vld')

w.append_data(image1)
w.append_data(image2)
w.append_data(image3)
w.close()