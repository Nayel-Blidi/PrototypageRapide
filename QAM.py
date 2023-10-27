import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm

class SixteenQAM():
    """
    A class dedicated to binary signal emission simulation in temporal resolution.
    16QAM stands for 16 (values) Quadrature Amplitude Modulation : 
        1) Every word in bit is transmitted over a duration 1/Rb as a cosin of frequency Fc
        2) Cosine amplitude is modulated by A in [0.25, 0.5, 0.75, 1] (In-Phase channel)
        3) Cosine phase is modulated by n * pi, n in [0.25, 0.5, 0.75, 1] (Quadrature channel)
    """

    def __init__(self, 
                 message=None,
                 word_length=4, words_number=10, 
                 sigma=1,
                 Fc=2e3, Fs=10e3, Rb=2e3,
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
        inphase_vector = np.round(np.linspace(-1, 1, self.word_length), 2)
        quadrature_vector = np.round(np.linspace(-1, 1, self.word_length), 2)

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
    
    def signalOversampling(self):
        """
        Method \n
        Oversamples the signal to introduce bit redundancy
        """
        self

    def signalModulation(self):
        """
        Method \n
        Modulates the keyed message into its temporal resolution version 
        """
        Fc = self.Fc
        Rb = self.Rb
        T = self.timeResolution
        t = np.linspace(0, T, T) # Time vector, timeResolution is a simulation parameter

        temporal_message = np.zeros((self.words_number, self.word_length*T))
        for row_idx, row in enumerate(self.message):
            for col_idx, col in enumerate(row):
                temporal_message[row_idx, col_idx*T:(col_idx+1)*T] = col * np.cos(2 * np.pi * (Fc/Rb) * t)   

        self.message = temporal_message   
        self.modulation = self.message
        return self.message
    
    def gaussianNoise(self):
        """
        Method \n
        Noises the modulated signal, as if it was emitted through an Additive White Noise Channel (AWGN)
        """
        self.message = self.message + np.random.rand(*self.message.shape) * self.sigma
        self.noised = self.message
        return self.message
    
    def stepVisualization(self, max_itt=2, show=True):
        """ 
        Function \n
        Plots the different steps of the BPSK process
        """
        plotLegend = False
        if self.word_length<8 and self.words_number<4:
            plotLegend = True

        if max_itt>4 or max_itt<0:
            max_itt = 4

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
            itt=0
            plt.subplots(max_itt, 1)
            while itt<max_itt:
                plt.subplot(max_itt, 1, itt+1)
                plt.plot(self.modulation[itt, :], label=f"word={self.sequence[itt,:]}")
                if plotLegend:
                    plt.legend(loc="upper right")
                itt+=1
            plt.suptitle(f"Temporal resolution modulated signal")
        except:
            error_message = "No modulated signal to plot. Verify the signalModulation() step."
            print(error_message)
        
        # %% Noisy signal
        try:
            itt=0
            plt.subplots(max_itt, 1)
            while itt<max_itt:
                plt.subplot(max_itt, 1, itt+1)
                plt.plot(self.noised[itt, :], label=f"word={self.sequence[itt,:]}")
                if plotLegend:
                    plt.legend(loc="upper right")
                itt+=1
            plt.suptitle(f"Noisy signal")
        except:
            error_message = "No noised signal to plot. Verify the gaussianNoise() step."
            print(error_message)

        # %% Time resolution signal
        try:
            itt=0
            plt.subplots(max_itt, 1)
            while itt<max_itt:
                plt.subplot(max_itt, 1, itt+1)
                plt.plot(self.sampled[itt, :], label=f"word={self.sequence[itt,:]}")
                if plotLegend:
                    plt.legend(loc="upper right")
                itt+=1
            plt.suptitle(f"Sampled signal")
        except:
            error_message = "No modulated signal to plot. Verify the signalSampling() step."
            print(error_message)

        # %% Decided message
        try:
            plt.subplots(2, 1)
            message_encoded = self.bpsk.flatten()
            message_decoded = self.decided.flatten()
            N = len(message_decoded)
            plt.subplot(2, 1, 1)
            plt.bar(np.linspace(0, N, N), message_encoded, label=f"message encoded={message_encoded}")
            plt.title("Encoded message")
            if plotLegend:
                plt.legend(loc="upper right")
            plt.subplot(2, 1, 2)
            plt.bar(np.linspace(0, N, N), message_decoded, label=f"message decoded={message_decoded}")
            plt.title("Decoded message")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No decided/bpsk signal to plot. Verify decided()/bpsk() step."
            print(error_message)

        # %%
        if not show and not self.visualizations:
            plt.close('all')
        
        return plt.show()


sixteen_qam = SixteenQAM(word_length=4, words_number=100)
message = sixteen_qam.sequenceGenerator()
symbols = sixteen_qam.QAM()
sixteen_qam.stepVisualization(show=True)