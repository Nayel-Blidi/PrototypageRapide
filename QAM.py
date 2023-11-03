import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
import sys

class SixteenQAM():
    """
    A class dedicated to binary signal emission simulation in temporal resolution.
    16QAM stands for 16 (values) Quadrature Amplitude Modulation : \n
        1) Every word in bit is transmitted over a duration 1/Rb as a cosin of frequency Fc \n
        2) Cosine amplitude is modulated by A in [0.25, 0.5, 0.75, 1] (In-Phase channel) \n
        3) Cosine phase is modulated by n * pi, n in [0.25, 0.5, 0.75, 1] (Quadrature channel) \n
    """

    def __init__(self, 
                 message=None,
                 word_length=4, words_number=10, 
                 sigma=1,
                 Fc=4, Fs=40, Rb=1,
                 timeResolution=100,
                 visualizations=False,
                 ):
        
        self.message = message
        self.word_length = word_length
        self.words_number = words_number
        self.sigma = sigma

        # We take a bit ratio equal to the carrier frequency for simplicity. If we want to tweak any of these parameters,
        # it can be acheived by oversampling the message (lower Rb, higher Fc) or cutting the signal (lower Fc, higher Rb)
        # Lowering the bit ratio makes it more robust (longer pattern for one bit) = oversampling, 
        # the same bit is repeted Fc/Rb times, and then the next bit is emitted by the carrier
        self.Fc = Fc # Carrier frequency
        self.Rb = Rb # Bit frequency (ratio)
        self.Fs = Fs # Sampling frequency
        self.Tc = 1/Fc
        self.Ts = 1/Fs
        
        self.timeResolution = timeResolution # Simulation parameter, unreal. Higher returns more detailled plots and reduces performances.
        self.visualizations = visualizations
        self._message = self.message

        if word_length != 4:
            print("WARNING: 16AM modulation is intended for 4 bits long words")

    def sequenceGenerator(self):
        """
        Method \n
        Generates a binary sequence according to the class parameters
        """
        self.message = np.round(np.random.rand(self.words_number, self.word_length)).astype(int)
        self.sequence = self.message
        return self.message

    def grayMapping(self):
        """
        Method \n
        Maps the class sequence using Gray mapping
        """
        self.message = self.message
        self.mapping = self.message
        return self.message
    
    def QAM(self):
        """
        Method \n
        Places the words in the complex plan, by attributing in-phase and quadrature symbols (positions)
        """
        m, n = np.shape(self.message)
        self.inphase = self.message[:, 0:n//2]
        self.quadrature = self.message[:, n//2:n]

        # Every binary word will be placed on the complex plan
        # The first half gives the in-phase (x-axis) position aranged between -1 and 1
        # The second half gives the quadrature (y-axis) position aranged between -1 and 1
        inphase_vector = np.round(np.linspace(-1+1/self.word_length, 1-1/self.word_length, self.word_length), 2)
        quadrature_vector = np.round(np.linspace(1/self.word_length, 1, self.word_length), 2)

        self.inphase_symbol = np.zeros((m)) 
        for row_idx, row in enumerate(self.inphase):
            # If bit value is 0, then symbol is -1, if bit value is 1, then symbol is -0.33 (for 4 arry symbols)
            self.inphase_symbol[row_idx] = inphase_vector[np.packbits(row, bitorder="little")]

        self.quadrature_symbol = np.zeros((m)) 
        for row_idx, row in enumerate(self.quadrature):
            # If bit value is 0, then symbol is -1, if bit value is 1, then symbol is -0.33 (for 4 arry symbols)
            self.quadrature_symbol[row_idx] = quadrature_vector[np.packbits(row, bitorder="little")]

        self.message = list(zip(self.inphase_symbol, self.quadrature_symbol))
        self.symbols = self.message
        return self.message 


    def signalModulation(self):
        """
        Method \n
        Modulates the keyed message into its temporal resolution version 
        """
        Fc = self.Fc
        Rb = self.Rb
        T = self.timeResolution
        t = np.linspace(0, 1/Rb, T) # Time vector, timeResolution is a simulation parameter

        temporal_message = np.zeros((self.words_number, T))
        modulated_message = np.zeros((self.words_number, T))
        for row_idx, row in enumerate(self.message):
            inphase_symbol, quadrature_symbol = self.message[row_idx]
            temporal_message[row_idx, 0:T] = quadrature_symbol * np.cos(2 * np.pi * Rb * t + np.pi*inphase_symbol)   
            modulated_message[row_idx, 0:T] = quadrature_symbol * np.cos(2 * np.pi * Fc * t + np.pi*inphase_symbol)   

        self.message = modulated_message
        self.temporal_message = temporal_message   
        self.modulated_message = modulated_message
        return self.message
    
    def gaussianNoise(self):
        """
        Method \n
        Noises the modulated signal, as if it was emitted through an Additive White Noise Channel (AWGN)
        """
        self.message = self.message + np.random.rand(*self.message.shape) * self.sigma
        self.noised = self.message
        return self.message

    def signalSampling(self):
        """
        Method \n
        Samples the temporal signal, as if received by an oscillator
        """
        T = self.timeResolution
        Fs = self.Fs
        Fc = self.Fc
        Rb = self.Rb
        # A value is retreived every step, which results in Fs values every second
        step = max(round(T/(Fs/Rb)), 1)

        temporal_message = np.zeros((self.words_number, T))
        for row_idx, row in enumerate(self.message):
            for col_idx in range(0, len(row), step):
                temporal_message[row_idx, col_idx:col_idx+step] = row[col_idx]

        self.message = temporal_message
        self.sampled = temporal_message
        return self.message
    
    def stepVisualization(self, show=True):
        """ 
        Function \n
        Plots the different steps of the BPSK process
        """
        plotLegend = True

        # %% Symbols plot
        try:
            plt.subplots(2, 1)
            plt.suptitle("Symbols sequence")
            N = len(self.inphase_symbol)
            plt.subplot(2, 1, 1)
            plt.bar(np.linspace(0, N, N), self.inphase_symbol)
            plt.title("In-phase channel")
            plt.subplot(2, 1, 2)
            plt.bar(np.linspace(0, N, N), self.quadrature_symbol)
            plt.title("Quadrature channel")
        except:
            error_message = "No symbol sequences. Verify symbolGenerator() step."
            print(error_message)

        # %% Symbols constellation
        try:
            plt.figure()
            plt.scatter(self.inphase_symbol, self.quadrature_symbol)
            plt.suptitle("Mapped symbols constellation")
            plt.grid(True)
        except:
            error_message = "No symbols constellation to plot. Verify symbolGenerator() step."
            print(error_message)

        # %% Time resolution signal
        try:
            plt.subplots(4, 1)
            plt.suptitle(f"Temporal resolution modulated signal")
            plt.subplot(4, 1, 1)
            plt.plot(self.temporal_message[0, :], label=f"simulated_message, bits={self.sequence[0,:]}")
            if plotLegend:
                plt.legend(loc="upper right")
            plt.subplot(4, 1, 2)
            plt.plot(self.modulated_message[0, :], label=f"modulated_message, bits={self.sequence[0,:]}")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No modulated signal to plot. Verify the signalModulation() step."
            print(error_message)
        
        # %% Noisy signal
        try:
            plt.subplot(4, 1, 3)
            plt.plot(self.noised[0, :], label=f"noised_message, bits={self.sequence[0,:]}")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No noised signal to plot. Verify the gaussianNoise() step."
            print(error_message)

        # %% Sampled signal
        try:
            plt.subplot(4, 1, 4)
            plt.plot(self.sampled[0, :], label=f"sampled_signal, bits={self.sequence[0,:]}")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No modulated signal to plot. Verify the signalSampling() step."
            print(error_message)

        # %%
        if not show and not self.visualizations:
            plt.close('all')
        
        return plt.show()


class SixteenQAM_light():
    """
    A higher performance version of SixteenQAM(), where only the sampled values are computed
    """

    def __init__(self, 
                 message=None,
                 word_length=4, words_number=10, 
                 sigma=1,
                 Fc=4, Fs=40, Rb=1,
                 timeResolution=100,
                 visualizations=False,
                 ):
        
        self.message = message
        self.word_length = word_length
        self.words_number = words_number
        self.sigma = sigma

        # We take a bit ratio equal to the carrier frequency for simplicity. If we want to tweak any of these parameters,
        # it can be acheived by oversampling the message (lower Rb, higher Fc) or cutting the signal (lower Fc, higher Rb)
        # Lowering the bit ratio makes it more robust (longer pattern for one bit) = oversampling, 
        # the same bit is repeted Fc/Rb times, and then the next bit is emitted by the carrier
        self.Fc = Fc # Carrier frequency
        self.Rb = Rb # Bit frequency (ratio)
        self.Fs = Fs # Sampling frequency
        self.Tc = 1/Fc
        self.Ts = 1/Fs
        
        self.timeResolution = timeResolution # Simulation parameter, unreal. Higher returns more detailled plots and reduces performances.
        self.visualizations = visualizations
        self._message = self.message

        if word_length != 4:
            print("WARNING: 16AM modulation is intended for 4 bits long words")


    def sequenceGenerator(self):
        """
        Method \n
        Generates a binary sequence according to the class parameters, and attributes the corresponding symbols
        in the complex plan 
        """
        self.message = np.round(np.random.rand(self.words_number, self.word_length)).astype(int)
        self.sequence = self.message

        m, n = np.shape(self.message)
        self.inphase = self.message[:, 0:n//2]
        self.quadrature = self.message[:, n//2:n]

        # Every binary word will be placed on the complex plan
        # The first half gives the in-phase (x-axis) position aranged between -1 and 1
        # The second half gives the quadrature (y-axis) position aranged between -1 and 1
        inphase_vector = np.round(np.linspace(-1+1/self.word_length, 1-1/self.word_length, self.word_length), 2)
        quadrature_vector = np.round(np.linspace(1/self.word_length, 1, self.word_length), 2)

        self.inphase_symbol = np.zeros((m)) 
        for row_idx, row in enumerate(self.inphase):
            # If bit value is 0, then symbol is -1, if bit value is 1, then symbol is -0.33 (for 4 arry symbols)
            self.inphase_symbol[row_idx] = inphase_vector[np.packbits(row, bitorder="little")]

        self.quadrature_symbol = np.zeros((m)) 
        for row_idx, row in enumerate(self.quadrature):
            # If bit value is 0, then symbol is -1, if bit value is 1, then symbol is -0.33 (for 4 arry symbols)
            self.quadrature_symbol[row_idx] = quadrature_vector[np.packbits(row, bitorder="little")]

        self.message = list(zip(self.inphase_symbol, self.quadrature_symbol))
        self.symbols = self.message
    
        return self.sequence


    def signalModulation(self):
        """
        Method \n
        Modulates the keyed message into its temporal resolution version 
        """
        Fc = self.Fc
        Rb = self.Rb
        Fs = self.Fs
        T = round(Fs//Rb)
        t = np.linspace(0, 1/Rb, T) # Time vector, timeResolution is a simulation parameter

        # temporal_message = np.zeros((self.words_number, T))
        modulated_message = np.zeros((self.words_number, T))
        for row_idx, row in enumerate(self.message):
            inphase_symbol, quadrature_symbol = row
            # temporal_message[row_idx, :] = quadrature_symbol * np.cos(2 * np.pi * Rb * t + np.pi*inphase_symbol)   
            modulated_message[row_idx, :] = quadrature_symbol * np.cos(2 * np.pi * Fc * t + np.pi*inphase_symbol)   

        self.message = modulated_message   
        self.modulated_message = modulated_message
        return self.message
    

    def gaussianNoise(self):
        """
        Method \n
        Noises the modulated signal, as if it was emitted through an Additive White Noise Channel (AWGN)
        """
        self.message = self.message + np.random.rand(*self.message.shape) * self.sigma
        self.noised = self.message
        return self.message

if __name__ =="__main__" and "16qam" in sys.argv:
    sixteen_qam = SixteenQAM(words_number=100)
    message = sixteen_qam.sequenceGenerator()
    symbols = sixteen_qam.QAM()
    qam = sixteen_qam.signalModulation()
    qam_n = sixteen_qam.gaussianNoise()
    sam_sampled = sixteen_qam.signalSampling()

    if __name__ =="__main__" and "plots" in sys.argv:
        sixteen_qam.stepVisualization(show=True)

if __name__ =="__main__" and ("16qam_light" in sys.argv) or ("light" in sys.argv):
    sixteen_qam_light = SixteenQAM_light(words_number=100)
    Y = sixteen_qam_light.sequenceGenerator()
    X_sim = sixteen_qam_light.signalModulation()
    X = sixteen_qam_light.gaussianNoise()

    if __name__ == "__main__" and "plots" in sys.argv:
        plt.subplots(2, 1)
        plt.suptitle("SixteenQAM_light simulation")
        plt.subplot(2, 1, 1)
        plt.plot(X_sim[0,:], label=f"{Y[0]}")
        plt.plot(X_sim[1,:], label=f"{Y[1]}")
        plt.plot(X_sim[2,:], label=f"{Y[2]}")
        plt.plot(X_sim[3,:], label=f"{Y[3]}")
        plt.legend(loc="upper right")
        plt.subplot(2, 1, 2)
        plt.plot(X[0,:], label=f"{Y[0]}")
        plt.plot(X[1,:], label=f"{Y[1]}")
        plt.plot(X[2,:], label=f"{Y[2]}")
        plt.plot(X[3,:], label=f"{Y[3]}")
        plt.legend(loc="upper right")
        plt.show()

if __name__ == "__main__" and "plots" in sys.argv:
    plt.subplots(2, 1)
    plt.subplot(2, 1, 1)
    for k in range(100):
        plt.plot(qam[k,:])
    plt.subplot(2, 1, 2)
    for k in range(10):
        plt.plot(qam_n[k,:])

    # plt.subplots(2, 1)
    # plt.subplot(2, 1, 1)
    # for k in range(4):
    #     plt.plot(qam[k,:])
    # plt.subplot(2, 1, 2)
    # for k in range(4):
    #     plt.plot(qam_n[k,:])
    plt.show()

