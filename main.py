import numpy as np
import QAM
import PSK
import mainDeepPSK as mainDPSK
import signal_handler as sh
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = base_dir + "/Image_inputs/"
print(image_dir)

file_list = os.listdir(image_dir)
image_list = []
image_list = [file for file in file_list if file.endswith(".jpg")]
image_list = image_list + [file for file in file_list if file.endswith(".png")]

for image_name in image_list:
    img = np.array(Image.open(image_dir + image_name), dtype=np.uint)
    # plt.imshow(img)
    # plt.show()
    m_img, n_img, in_channels = img.shape
    print(m_img, n_img)

    img_to_bin_sg = sh.SignalHandler(img, is_bin=False).converted_signal
    m_sig, n_sig = img_to_bin_sg.shape
    print(m_sig, n_sig)

    img_to_bin_sg = np.array([img_to_bin_sg.flatten()]).T


    if "bpsk" in sys.argv:
        bpsk_pipeline = [
            PSK.BPSK.BPSK,
            PSK.BPSK.signalModulation,
            PSK.BPSK.gaussianNoise,
            PSK.BPSK.signalSampling,
            PSK.BPSK.sequenceDecider,
        ]

        words_length=1
        words_number=img_to_bin_sg.shape[0]
        timeResolution = 100
        bpsk_class = PSK.BPSK(message=img_to_bin_sg, word_length=words_length, words_number=words_number, 
                              sigma=1, timeResolution=timeResolution)
        for step in tqdm(bpsk_pipeline, desc="BPSK Pipeline progression"):
            message = step(bpsk_class)

        print(message.shape)
        correct_bit = np.sum(np.equal(img_to_bin_sg, message.T))
        total_bits = img_to_bin_sg.size
        print("success rate =", round(correct_bit/total_bits, 2))
        print("correct bits =", correct_bit, "total bits =", total_bits)

        message = np.reshape(message, (m_sig, n_sig))
        bit_to_img = sh.SignalHandler(sig=message, is_bin=True)
        bit_to_img.to_image(shape=[m_img, n_img, 3])
        image_received = bit_to_img.converted_signal
        print(image_received.shape)

        # plt.imshow(image_received)
        # plt.show()

        image = Image.fromarray(image_received)
        image.save(base_dir + "/Image_outputs/bpsk_" + str(timeResolution) +"_" + image_name)

    if "bpsk_nn" in sys.argv:
        bpsk_pipeline = [
            PSK.BPSK.BPSK,
            PSK.BPSK.signalModulation,
            PSK.BPSK.gaussianNoise,
            PSK.BPSK.signalSampling,
        ]

        words_length=1
        words_number=img_to_bin_sg.shape[0]
        timeResolution = 100
        bpsk_class = PSK.BPSK(message=img_to_bin_sg, word_length=words_length, words_number=words_number, 
                              sigma=1, timeResolution=timeResolution, Fs=8)
        input_size = round((bpsk_class.Fs / bpsk_class.Rb))
        for step in tqdm(bpsk_pipeline, desc="BPSK Pipeline progression"):
            message = step(bpsk_class)
        # print(message.shape)
        # step = round(bpsk_class.timeResolution/(bpsk_class.Fs/bpsk_class.Fc))
        # message = message[:, ::step]
        message = message[:, 0:input_size]
        print(message.shape, input_size)

        if ("LayeredNN" in sys.argv):
            num_epochs = int(input("Model's number of epochs to load : "))
            model = mainDPSK.LayeredNN(input_size=input_size, hidden_size=128)
            model.load_state_dict(torch.load(f"Models/BPSK_LayeredNN_{num_epochs}.pth"))

        if ("ConvNN" in sys.argv):
            num_epochs = int(input("Model's number of epochs to load : "))
            model = mainDPSK.ConvNN(input_size=input_size, hidden_size=128)
            model.load_state_dict(torch.load(f"Models/BPSK_ConvNN_{num_epochs}.pth"))

        if ("SequentialNN" in sys.argv):
            num_epochs = int(input("Model's number of epochs to load : "))
            model = mainDPSK.SequentialLayeredNN(input_size=input_size, hidden_size=128)
            model.load_state_dict(torch.load(f"Models/BPSK_SequentialLayeredNN_{num_epochs}.pth"))
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            test_features = torch.tensor(message).float()
            if "ConvNN" in sys.argv:
                test_features = torch.unsqueeze(input=test_features, dim=1)

            outputs = model(test_features)
            print(outputs[0:5, :])
            predicted = torch.round(outputs.data).numpy()
        
        print(predicted[0:10])
        print(predicted.shape)

        message = np.reshape(predicted, (m_sig, n_sig))
        print(message.shape)
        bit_to_img = sh.SignalHandler(sig=message, is_bin=True)
        bit_to_img.to_image(shape=[m_img, n_img, 3])
        image_received = bit_to_img.converted_signal
        print(image_received.shape)

        image = Image.fromarray(image_received)
        image.save(base_dir + "/Image_outputs/bpsk_nn_" + str(timeResolution) +"_" + image_name)


    if "qam" in sys.argv:
        None


    break

