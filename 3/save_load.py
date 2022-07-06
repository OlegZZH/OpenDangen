import numpy as np
def save_patch(point):
    np.savez("surface1", patch1=point[:, :, :3][1::2], patch2=point[:, :, :3][::2])


def load_patch():
    load_data = np.load("surface1.npz")
    print(load_data["patch2"])
    return load_data