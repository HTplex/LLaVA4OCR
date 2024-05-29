import numpy as np
import matplotlib.pyplot as plt
def normalize_landmarks(landmarks):
    # normalize landmarks to 0-1 x,y,z
    # input, batch_size,478,3
    # normalize landmark to 0-1 x,y,z
    landmarks[:,:,0] = (landmarks[:,:,0] - landmarks[:,:,0].min())/(landmarks[:,:,0].max()-landmarks[:,:,0].min())
    landmarks[:,:,1] = (landmarks[:,:,1] - landmarks[:,:,1].min())/(landmarks[:,:,1].max()-landmarks[:,:,1].min())
    landmarks[:,:,2] = (landmarks[:,:,2] - landmarks[:,:,2].min())/(landmarks[:,:,2].max()-landmarks[:,:,2].min())
    # print(np.min(landmarks[:,:,0],axis=1))
    return landmarks


def show_img_np(img, max_h=3, max_w=20, save=False, cmap='gray'):
    """
    :param np_array: input image, one channel or 3 channel,
    :param save: if save image
    :param size:
    :return:
    """
    if len(img.shape) < 3:
        plt.rcParams['image.cmap'] = cmap
    plt.figure(figsize=(max_w, max_h), facecolor='w', edgecolor='k')
    plt.imshow(img)
    if save:
        cv2.imwrite('debug.png', img)
    else:
        plt.show()
        