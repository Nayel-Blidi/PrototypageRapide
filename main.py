import numpy as np
import QAM
import PSK
import signal_handler as sh
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = base_dir + "/Image/"
print(image_dir)

file_list = os.listdir(image_dir)
image_list = []
image_list = [file for file in file_list if file.endswith(".jpg")]
image_list = image_list + [file for file in file_list if file.endswith(".png")]

for image_path in image_list:
    img = np.array(Image.open(image_dir + image_path), dtype=np.uint)
    plt.imshow(img)
    m_img, n_img, in_channels = img.shape
    print(m_img, n_img)

    img_to_bin_sg = sh.SignalHandler(img, is_bin=False).converted_signal
    m_sig, n_sig = img_to_bin_sg.shape
    print(m_sig, n_sig)

    img_to_bin_sg = np.array([img_to_bin_sg.flatten()]).T


    if "bpsk" in sys.argv:
        bpsk_pipeline = [
            # PSK.BPSK.grayMapping,
            PSK.BPSK.BPSK,
            PSK.BPSK.signalModulation,
            PSK.BPSK.gaussianNoise,
            PSK.BPSK.signalSampling,
            PSK.BPSK.sequenceDecider,
            # PSK.BPSK.grayDemapping,
        ]

        bpsk_light_pipeline = [
            PSK.BPSK_light.signalModulation,
            PSK.BPSK_light.gaussianNoise,
        ]

        # if "light" in sys.argv:
        #     words_length=1
        #     words_number=img_to_bin_sg.shape[0]
        #     bpsk_class = PSK.BPSK_light(message=img_to_bin_sg, word_length=words_length, words_number=words_number, sigma=1)
        #     for step in tqdm(bpsk_light_pipeline, desc="BPSK_light Pipeline progression"):
        #         message = step(bpsk_class)
        
        words_length=1
        words_number=img_to_bin_sg.shape[0]
        bpsk_class = PSK.BPSK(message=img_to_bin_sg, word_length=words_length, words_number=words_number, sigma=1, timeResolution=10)
        for step in tqdm(bpsk_pipeline, desc="BPSK Pipeline progression"):
            message = step(bpsk_class)

        correct_bit = np.sum(np.equal(img_to_bin_sg, message.T))
        total_bits = img_to_bin_sg.size
        print("success rate =", round(correct_bit/total_bits, 2))
        print("correct bits =", correct_bit, "total bits =", total_bits)

        message = np.reshape(message, (m_sig, n_sig))
        bit_to_img = sh.SignalHandler(sig=message, is_bin=True)
        bit_to_img.to_image(shape=[m_sig, n_sig, 3])
        print(bit_to_img.shape)

        plt.imshow(bit_to_img)

    if "qam" in sys.argv:
        None



    break