import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from PIL import Image
from sklearn.preprocessing import minmax_scale
from keras.models import Model

def features_heatmap(model, input_data, layer_name, stack_by=8):

    cmap = mpl.cm.get_cmap('jet')

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)
    x = input_data
    a = intermediate_output

    x = np.swapaxes(x[0,:,:,:],0,-1)
    a = np.swapaxes(a[0,:,:,:],0,-1) 

    acts = minmax_scale(a.reshape(-1, a.shape[-1]), feature_range=(0, 1), axis=1, copy=True)
    acts = acts.reshape(a.shape)

    img_data = x.astype(np.uint8)
    width,height,_ = img_data.shape
    img = Image.fromarray(img_data, mode='RGBA')
    _, _, features = acts.shape

    lst = []
    for i in range(features):
        hm = acts[:, :, i]
        hm = Image.fromarray(hm, mode='F')

        hm = hm.resize((width, height), Image.LANCZOS)
        hm = (cmap(np.array(hm)) * 255).astype(np.uint8)
        hm = Image.fromarray(hm, mode='RGBA')
        
        blend = Image.blend(img, hm, alpha=0.5)
        lst.append(np.array(blend))

    final_img = np.vstack([np.hstack(lst[i:i+stack_by]) for i in range(0, len(lst), stack_by)])
    return final_img
