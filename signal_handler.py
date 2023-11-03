import numpy as np


class SignalHandler(object):
    original_signal: int | np.ndarray[int]
    converted_signal: int | np.ndarray[int]
    is_bin: bool

    def __init__(self, sig: int | np.ndarray[int], is_bin: bool):
        self.original_signal = sig
        self.is_bin = is_bin

        # Convert binary signal (array of shape (n, 8)) to int array
        if is_bin:
            self.bin_to_any(sig)
        # Convert int or array into binary signal (array of shape (n, 8))
        else:
            if isinstance(sig, int):
                self.int_to_bin(sig)
            if isinstance(sig, np.ndarray):
                self.img_to_bin(sig)

    def __str__(self) -> str:
        tmp = ""
        if isinstance(self.converted_signal, int):
            tmp = str(self.converted_signal) + "\n"

        if isinstance(self.converted_signal, np.ndarray):
            if self.is_bin:
                tmp += str(self.converted_signal) + "\n"
            else:
                for sub_signal in self.converted_signal:
                    tmp += str(sub_signal) + "\n"
        return tmp

    def int_to_bin(self, sig: int, nb_bits: int = 8) -> None:
        """
        Convert int value into binary signal of 8 bits
        """
        lst = list(bin(sig)[2:].zfill(nb_bits))
        self.converted_signal = np.array([[int(e) for e in lst]], dtype=int)
        try:
            assert len(lst) == nb_bits
        except AssertionError:
            raise Exception("Number of bits exceeded")

    def bin_to_any(self, sig: np.ndarray[int]) -> None:
        """
        Convert binary signal of 8 bits into int array
        """
        lst = []
        for sub_signal in sig:
            sub = int("".join([str(e) for e in sub_signal]), 2)
            lst.append(sub)
        self.converted_signal = np.array(lst, dtype=np.uint8)

    def img_to_bin(self, sig: np.ndarray) -> None:
        """
        Convert image (array) into array of binary signals
        """
        sig_flat = list(sig.flatten())
        lst = [[int(elt) for elt in bin(e)[2:].zfill(8)] for e in sig_flat]
        self.converted_signal = np.array(lst, dtype=int)

    def to_image(self, shape: tuple[int, int, int]) -> None:
        """
        Reshape array
        """
        self.converted_signal = self.converted_signal.reshape(shape)
