import numpy as np
import re
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import join
import cv2

def data_gen_sobel(path, ids):
    """
    In this generator every input is interesected with the Prostate contour
    :param path:
    :param folders_to_read:
    :param img_names:
    :param roi_names:
    :param tot_ex_per_img
    :return:
    """
    print( "Inside generator...")

    curr_idx = -1 # First index to use

    all_files_per_digit = []
    tot_files_per_digit = []
    acc_files_per_digit = []
    last_id = 0
    for cur_folder in range(10):
        all_files = listdir(join(path, F'{cur_folder}'))
        tot_files = len(all_files)
        last_id += tot_files
        all_files_per_digit.append(all_files)
        tot_files_per_digit.append(tot_files)
        acc_files_per_digit.append(last_id)

    tot_files_per_digit = np.array(tot_files_per_digit)
    acc_files_per_digit = np.array(acc_files_per_digit)
    all_files_per_digit = np.array(all_files_per_digit)

    tot_files = acc_files_per_digit[-1]
    print(tot_files_per_digit)
    print(acc_files_per_digit)

    while True:
        # These lines are for sequential selection
        if curr_idx >= len(ids) or curr_idx == -1:
            curr_idx = 0
            np.random.shuffle(ids) # We shuffle the folders every time we have tested all the examples
        else:
            curr_idx += 1

        try:
            cur_file = ids[curr_idx]
            cur_folder = np.argmax(acc_files_per_digit >= cur_file)
            if cur_folder > 0:
                folder_idx = cur_file - acc_files_per_digit[cur_folder]
            else:
                folder_idx = cur_file

            file_name = join(path, F'{cur_folder}', F'{all_files_per_digit[cur_folder][folder_idx]}')
            X_rgb = cv2.imread(file_name)
            X = X_rgb[:,:,0]
            Y = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=3)

            # plt.subplots(1,2)
            # plt.subplot(1,2,1)
            # plt.imshow(X)
            # plt.subplot(1,2,2)
            # plt.imshow(Y)
            # plt.show()

            # Normalizing the input
            X = X/255
            max_val = 255*4
            Y = (Y+max_val)/(max_val*2)

            XF = np.expand_dims(np.expand_dims(X, axis=2), axis=0)
            YF = np.expand_dims(np.expand_dims(Y, axis=2), axis=0)
            yield XF, YF
        except Exception as e:
            print(F"----- Not able to generate for curr_idx: {curr_idx}, file_name: {file_name}")


# if __name__ == "__main__":
#     data_gen_sobel('/home/olmozavala/Dropbox/TestData/MNIST/training', np.arange(0,8000))
