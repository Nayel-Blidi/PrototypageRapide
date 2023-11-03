import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from signal_handler import SignalHandler

# Convert int to binary signal
int_sig = 42
int_to_bin_sg = SignalHandler(int_sig, is_bin=False)
print(int_to_bin_sg)

# Convert image to array of binary signals
base_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(base_dir, "Images/chat")
img = Image.open(file)
img_sig = np.array(img, dtype=np.uint8)
h, w, c = img_sig.shape
print(h * w * c)
print(img_sig)

#img_sig = np.array([
#    [[100, 100, 100], [10, 0, 255]],
#    [[1, 100, 255], [0, 255, 0]],
#], dtype=np.uint8)

img_to_bin_sg = SignalHandler(img_sig, is_bin=False)
#print(img_to_bin_sg)

# Convert array of binary signals into array of int
bin_sig = np.array([[0, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]])
bin_to_int_sg = SignalHandler(bin_sig, is_bin=True)
#print(bin_to_int_sg)

# Convert binary signal into reshaped image
sg_to_img = SignalHandler(img_to_bin_sg.converted_signal, is_bin=True)
sg_to_img.to_image(shape=(h, w, c))
#print(sg_to_img)


matplotlib.use("TkAgg")
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(img_sig)
ax[0].set_title('Original Image')
ax[1].imshow(sg_to_img.converted_signal)
ax[1].set_title('Binary signal to Image')
plt.show()
