import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
import sys
from scipy import signal
from scipy.signal import hilbert

"""
En BPSK, pour chaque unité de transmission (Tc=1/Fc carrier frequency), on transmet 1 valeur à 2 possibilités, 0 ou 1
En QPSK, 1 valeur à 4 possibilités : 00, 01, 10, 11
En 4QAM = QPSK (aka 4PSK)
En 16QAM, 16 possibilités : 0000, 0001...
"""
# %% BPSK
class BPSK():
    """
    A class dedicated to binary signal emission simulation in temporal resolution.
    BPSK stands for Binary Phase Key Shiftting : 
        1) Every bit is transmitted over a duration 1/Rb as a cosin of frequency Fc
        2) Cosine phase is modulated by n in [0.5, 1]
    """

    def __init__(self, 
                 message=None,
                 word_length=1, words_number=2, 
                 sigma=1,
                 Fc=4e2, Fs=20e2, Rb=1e2,
                 timeResolution=10e2,
                 visualizations=False,
                 ):
        
        self.message = message
        self.word_length = round(word_length)
        self.words_number = round(words_number)
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

        if word_length != 1:
            print("WARNING: BPSK modulation is intended for 1 bit long words.")


    def _LowPassFilter(self, modulated_message, t):
        filtre =  np.cos(2 * np.pi * self.Rb * t) #/ (2 * np.pi * t)
        demod = modulated_message * 2 * np.cos(2*np.pi*self.Fc*t)
        return np.convolve(filtre, demod)

    def _ButterwothFilter(self, modulated_message):
        b, a = signal.butter(N=2, Wn=0.1, btype='low')
        return signal.filtfilt(b, a, modulated_message)
    

    def sequenceGenerator(self):
        """
        Method \n
        Generates a binary sequence according to the class parameters
        """
        self.message = np.round(np.random.rand(self.words_number, self.word_length)).astype(np.int32)
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
    

    def BPSK(self):
        """
        Method \n
        Returns the bit message keyed using bpsk algorithm 
        """
        self.message = 2 * self.message - 1
        self.bpsk = self.message
        return self.message
    

    def signalOversampling(self):
        """
        Method \n
        Oversamples the signal to introduce bit redundancy
        """
        pass


    def signalModulation(self):
        """
        Method \n
        Modulates the keyed message into its temporal resolution version 
        """
        Fc = self.Fc
        Rb = self.Rb
        T = round(self.timeResolution)
        t = np.linspace(0, 1/Rb, T) # Time vector, timeResolution is a simulation parameter

        temporal_message = np.zeros((self.words_number, self.word_length*T))
        modulated_message = np.zeros((self.words_number, self.word_length*T))
        for row_idx, row in enumerate(self.message):
            for col_idx, col in enumerate(row):
                temporal_message[row_idx, col_idx*T:(col_idx+1)*T] = col * np.cos(2 * np.pi * t * Rb)
                # modulated_message[row_idx, col_idx*T:(col_idx+1)*T] = np.convolve(col *np.cos(2 * np.pi * t * Rb) , np.cos(2 * np.pi * Fc * t))[0:T]
                modulated_message[row_idx, col_idx*T:(col_idx+1)*T] = col *np.cos(2 * np.pi * t * Fc)

        self.temporal_message = temporal_message 
        self.modulated_message = modulated_message
        self.message = modulated_message
        return self.message
    

    def gaussianNoise(self):
        """
        Method \n
        Noises the modulated signal, as if it was emitted through an Additive White Noise Channel (AWGN)
        """
        message = self.message + np.random.rand(*self.message.shape) * self.sigma
        self.noised = message

        # T = round(self.timeResolution)
        # t = np.linspace(0, 1/self.Rb, T) # Time vector, timeResolution is a simulation parameter
        # noisy_message = np.zeros(message.shape)
        # for row_idx, row in enumerate(message):
        #     # self.message[row_idx, :] = self._ButterwothFilter(modulated_message=row)
        #     # self.message[row_idx, :] = np.angle((hilbert(row)))
        #     noisy_message[row_idx, :] = self._LowPassFilter(modulated_message=row, t=t)[0:T]

        # self.noised_demodulated = noisy_message
        # self.message = noisy_message
        return self.message
    

    def signalSampling(self):
        """
        Method \n
        Samples the temporal signal, as if received by an oscillator
        """
        T = round(self.timeResolution)
        Fs = self.Fs
        Fc = self.Fc
        Rb = self.Rb
        # A value is retreived every step, which results in Fs values every second (if possible, otherwise T values)
        step = max(round(T/(Fs/Rb)), 1)

        temporal_message = np.zeros((self.words_number, self.word_length*T))
        for row_idx, row in enumerate(self.message):
            for col_idx in range(0, len(row), step):
                    temporal_message[row_idx, col_idx:col_idx+step] = row[col_idx]

        self.sampled = temporal_message
        return self.message
    

    def sequenceDecider(self):
        """
        Method \n
        Decides the bit value of every received symbol
        """
        T = self.timeResolution
        Fs = self.Fs
        Fc = self.Fc
        # A value is retreived every step, which results in Fs values every second
        step = round(T/(Fs/Fc))

        self.message = self.message.flatten()#.astype(complex)
        self.message = self.message[::step]

        real_message = []
        imag_message = []
        for sample_idx, sample in enumerate(self.message):
            #if 5 samples correspond to a bit, then the cos is computed for 5 values between 0 and T
            local_progression = sample_idx%(Fs/Fc) / (Fs/Fc)
            real_message.append(sample * np.cos(2*np.pi*local_progression))
            imag_message.append(sample * np.sin(2*np.pi*local_progression))

        encoded_message = []
        constellation = []
        for bit_position in range(0, len(self.message), round(Fs/Fc)):
            y_real = real_message[bit_position : (bit_position+round(Fs/Fc))]
            y_imag = imag_message[bit_position : (bit_position+round(Fs/Fc))]
            real_integrate = integrate.trapz(y_real)
            imag_integrate = integrate.trapz(y_imag)

            constellation.append(real_integrate)
            constellation.append(imag_integrate)
            if real_integrate > imag_integrate:
                # If the curve correspond to a cos, then S(cos*cos) > S(cos*sin) (S the integrale)
                encoded_message.append(1)
            else:
                # If the curve correspond to a cos, then S(sin*sin) > S(cos*sin)
                encoded_message.append(-1)

        self.constellation = constellation
        self.message = np.array([encoded_message])
        self.decided = np.array([encoded_message])
        self.message[self.message == -1] = 0
        return self.message


    def grayDemapping(self):
        """
        Method \n
        Demaps the encoded sequence into standard binary encoding
        """
        self.message = self.message
        self.demapped = self.message
        return self.message


    def stepVisualization(self, max_itt=2, show=False):
        """ 
        Function \n
        Plots the different steps of the BPSK process
        """
        plotLegend = False
        if (self.word_length * self.words_number) <= 16:
            plotLegend = True

        if max_itt>4 or max_itt<0:
            max_itt = 4

        # %% Bit sequence plot
        plt.subplots(2, 1)
        plt.suptitle("Message sequence")
        try:
            message = self._message
            N = len(message)
            plt.subplot(2, 1, 1)
            plt.bar(np.linspace(0, N-1, N), message, label="Bits sequence")
            plt.title("Bits sequence")
        except:
            None
        try:
            message = self.sequence.flatten()
            N = len(message)
            plt.subplot(2, 1, 1)
            plt.bar(np.linspace(0, N-1, N), message, label="Bits sequence")
            plt.title("Bits sequence")
        except:
            error_message = "No message nor sequence to plot. initialization inputs or the sequenceGenerator() step"
            print(error_message)

        # %% BPSK sequence
        try:
            plt.subplot(2, 1, 2)
            message = self.bpsk.flatten()
            N = len(message)
            plt.bar(np.linspace(0, N-1, N), message, label=f"message={(message+1)//2}")
            plt.title("BPSK encoded sequence")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No BPSK signal to plot. Verify BPSK() step"
            print(error_message)
            
        # %% Time resolution signal
        try:
            plt.subplots(4, 1)
            plt.subplot(4, 1, 1)
            plt.plot(self.temporal_message[0, :], label=f"bit={self.sequence.flatten()[0]}")
            plt.title("temporal_message")
            plt.subplot(4, 1, 2)
            plt.plot(self.modulated_message[0, :], label=f"bit={self.sequence.flatten()[0]}")
            plt.title("modulated_message")

            if plotLegend:
                plt.legend(loc="upper right")
            plt.suptitle(f"Temporal resolution modulated signal")
        except:
            error_message = "No modulated signal to plot. Verify the signalModulation() step."
            print(error_message)
        
        # %% Noisy signal
        try:
            plt.subplot(4, 1, 3)
            plt.plot(self.noised[0, :], label=f"bit={self.sequence.flatten()[0]}")
            plt.title("noised_message")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No noised signal to plot. Verify the gaussianNoise() step."
            print(error_message)

        # %% Time resolution signal
        try:
            plt.subplot(4, 1, 4)
            plt.plot(self.sampled[0, :], label=f"bit={self.sequence.flatten()[0]}")
            plt.title(f"Sampled signal")
            if plotLegend:
                    plt.legend(loc="upper right")
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
            plt.bar(np.linspace(0, N, N), message_encoded, label=f"message encoded={message_encoded[:]}")
            plt.title("Encoded message")
            if plotLegend:
                plt.legend(loc="upper right")
            plt.subplot(2, 1, 2)
            plt.bar(np.linspace(0, N, N), message_decoded, label=f"message decoded={message_decoded[:]}")
            plt.title("Decoded message")
            if plotLegend:
                plt.legend(loc="upper right")
        except:
            error_message = "No decided/bpsk signal to plot. Verify decided()/bpsk() step"
            print(error_message)

        # %% Constellation
        try:
            plt.figure()
            constellation_real = self.constellation[0::2]
            constellation_imag = self.constellation[1::2]
            plt.scatter(constellation_real, constellation_imag)
            plt.grid(True)
            plt.suptitle("Decision constellation")
        except:
            error_message = "No constellation to plot. Verify decided() step"
            print(error_message)

        # %%
        if not show and not self.visualizations:
            plt.close('all')
        
        return plt.show()

# %%
import time
"""
But du NN : sauter l'étape sequenceDecider, voire  
"""

bpsk_pipeline = [
    BPSK.sequenceGenerator,
    BPSK.grayMapping,
    BPSK.BPSK,
    BPSK.signalModulation,
    BPSK.gaussianNoise,
    BPSK.signalSampling,
    BPSK.sequenceDecider,
    BPSK.grayDemapping,
]

bpsk_class = BPSK(word_length=1, words_number=12)
for step in tqdm(bpsk_pipeline, desc="BPSK Pipeline progression"):
    message = step(bpsk_class)
    time.sleep(0.1)
np.savetxt("BPSK_output.txt", message, delimiter=",", fmt="%d")

if __name__ == "__main__" and "plots" in sys.argv:
    bpsk_class.stepVisualization(show=True, max_itt=4)

if __name__ == "__main__" and "cos" in sys.argv:
    t = np.linspace(0, np.pi, 100)
    # plt.plot(t, np.sin(t), label="sin")
    # plt.plot(t, np.cos(t), label="cos")
    plt.plot(t, np.sin(t)**2, label="sin²")
    plt.plot(t, np.cos(t)**2, label="cos²")
    plt.plot(t, np.cos(t)*np.sin(t), label="cos*sin")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.figure()
    plt.plot(t, np.cos(t)*np.cos(t), label="np.cos(t)*np.cos(t)")
    plt.plot(t, np.cos(t)*np.cos(t+1), label="np.cos(t)*np.cos(t+1)")
    plt.plot(t, np.cos(t)*np.cos(t+2), label="np.cos(t)*np.cos(t+2)")
    plt.plot(t, np.cos(t)*np.cos(t+3), label="np.cos(t)*np.cos(t+3)")
    # plt.plot(t, np.cos(t)*np.sin(t), label="np.cos(t)*np.sin(t)")
    plt.title("In-phase signal integral is always larger than when out of phase")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()