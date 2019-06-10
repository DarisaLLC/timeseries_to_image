import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def series_to_img(seq, img_width=28):
    x_min, y_min = np.min(seq[:,0]), np.min(seq[:,1])
    x_max, y_max = np.max(seq[:,0]), np.max(seq[:,1])
    ratio = (y_max-y_min) / (x_max-x_min)

    seq[:, 0] = 0.8 * (seq[:, 0] - x_min) / (x_max - x_min)
    seq[:, 1] = 0.8 * (seq[:, 1] - y_min) / (y_max - y_min)

    if ratio > 1:
        seq[:, 0] /= ratio 
        seq[:, 0] += 0.1 + 0.4*(1-1/ratio)
        seq[:, 1] += 0.1
    else:
        seq[:, 1] *= ratio
        seq[:, 1] += 0.1 + 0.4*(1-ratio)
        seq[:, 0] += 0.1
    
    seq = seq * (img_width-1)
    seq = seq.astype(int)

    img = np.zeros((img_width, img_width), dtype=np.float32)
    
    for i in range(len(seq)):
        x, y = seq[i,0], seq[i,1]
        img[y,x] = 255.0

    return img

data_path = "data/CharacterTrajectories.npz"

series_data = np.load(data_path)

img_data = {}
img_data['x_train'] = [series_to_img(seq, 28) for seq in series_data['x_train']]
img_data['x_test'] = [series_to_img(seq, 28) for seq in series_data['x_test']]
img_data['y_train'] = series_data['y_train']
img_data['y_test'] = series_data['y_test']

np.savez("data/CharacterTrajectories_Image.npz", **img_data)




