from typing import Tuple, Sequence

import numpy as np
from scipy.signal import butter as bw, filtfilt, resample
from sklearn.base import BaseEstimator, TransformerMixin


def load_GIGA(
    db,
    sbj: int,
    eeg_ch_names: Sequence[str],
    fs: float,
    f_bank: np.ndarray,
    vwt: np.ndarray,
    new_fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function loads the GIGA-Science dataset locally.

    Parameters
    ----------
    db: GIGA_MI_ME
        A GIGA_MI_ME object created by the gcpds.databases.GIGA_MI_ME module
    sbj: int
        The subject to load
    eeg_ch_names: Sequence[str]
        The EEG channel names in order
    fs: float
        The sampling frecuency
    f_bank: np.ndarray
        The frecuency range(s) to use
    vwt: np.ndarray
        The time window to load
    new_fs: float
        The new sampling frecuency to resample the data to

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the EEG signals for each trial and the corresponding label

    Notes
    ----------
    The database description can be found here:
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323
    """
    index_eeg_chs = db.format_channels_selectors(channels=eeg_ch_names) - 1

    tf_repr = TimeFrequencyRpr(sfreq=fs, f_bank=f_bank, vwt=vwt)

    db.load_subject(sbj)
    X, y = db.get_data(
        classes=['left hand mi', 'right hand mi']
    )  # Load MI classes, all channels {EEG}, reject bad trials, uV
    X = X[:, index_eeg_chs, :]  # spatial rearrangement
    X = np.squeeze(tf_repr.transform(X))
    # Resampling
    if new_fs == fs:
        print('No resampling, since new sampling rate same.')
    else:
        print("Resampling from {:f} to {:f} Hz.".format(fs, new_fs))
        X = resample(X, int((X.shape[-1] / fs) * new_fs), axis=-1)

    # print(np.mean (X), np.var(X))
    return X, y


def butterworth_digital_filter(
    X,
    N,
    Wn,
    btype,
    fs,
    axis=-1,
    padtype=None,
    padlen=0,
    method='pad',
    irlen=None,
):
    """
    Apply digital butterworth filter
    INPUT
    ------
    1. X: (D array)
    array with signals.
    2. N: (int+)
    The order of the filter.
    3. Wn: (float+ or 1D array)
    The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for bandpass and bandstop filters, Wn is a length-2 vector.
    For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
    If fs is not specified, Wn units are normalized from 0 to 1, where 1 is the Nyquist frequency (Wn is thus in half cycles / sample and defined as 2*critical frequencies / fs). If fs is specified, Wn is in the same units as fs.
    4. btype: (str) {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    The type of filter
    5. fs: (float+)
    The sampling frequency of the digital system.
    6. axis: (int), Default=1.
    The axis of x to which the filter is applied.
    7. padtype: (str) or None, {'odd', 'even', 'constant'}
    This determines the type of extension to use for the padded signal to which the filter is applied. If padtype is None, no padding is used. The default is ‘odd’.
    8. padlen: (int+) or None, Default=0
    The number of elements by which to extend x at both ends of axis before applying the filter. This value must be less than x.shape[axis] - 1. padlen=0 implies no padding.
    9. method: (str), {'pad', 'gust'}
    Determines the method for handling the edges of the signal, either “pad” or “gust”. When method is “pad”, the signal is padded; the type of padding is determined by padtype
    and padlen, and irlen is ignored. When method is “gust”, Gustafsson’s method is used, and padtype and padlen are ignored.
    10. irlen: (int) or None, Default=nONE
    When method is “gust”, irlen specifies the length of the impulse response of the filter. If irlen is None, no part of the impulse response is ignored.
    For a long signal, specifying irlen can significantly improve the performance of the filter.
    OUTPUT
    ------
    X_fil: (D array)
    array with filtered signals.
    """
    b, a = bw(N, Wn, btype, analog=False, output='ba', fs=fs)
    return filtfilt(
        b,
        a,
        X,
        axis=axis,
        padtype=padtype,
        padlen=padlen,
        method=method,
        irlen=irlen,
    )


class TimeFrequencyRpr(BaseEstimator, TransformerMixin):
    """
    Time frequency representation of EEG signals.

    Parameters
    ----------
    1. sfreq:  (float) Sampling frequency in Hz.
    2. f_bank: (2D array) Filter banks Frequencies. Default=None
    3. vwt:    (2D array) Interest time windows. Default=None
    Methods
    -------
    1. fit(X, y=None)
    2. transform(X, y=None)
    """

    def __init__(self, sfreq, f_bank=None, vwt=None):
        self.sfreq = sfreq
        self.f_bank = f_bank
        self.vwt = vwt

    # ------------------------------------------------------------------------------

    def _validation_param(self):
        """
        Validate Time-Frequency characterization parameters.
        INPUT
        -----
          1. self
        ------
          2. None
        """
        if self.sfreq <= 0:
            raise ValueError('Non negative sampling frequency is accepted')

        if self.f_bank is None:
            self.flag_f_bank = False
        elif self.f_bank.ndim != 2:
            raise ValueError('Band frequencies have to be a 2D array')
        else:
            self.flag_f_bank = True

        if self.vwt is None:
            self.flag_vwt = False
        elif self.vwt.ndim != 2:
            raise ValueError('Time windows have to be a 2D array')
        else:
            self.flag_vwt = True

    # ------------------------------------------------------------------------------
    def _filter_bank(self, X):
        """
        Filter bank Characterization.
        INPUT
        -----
          1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
        OUTPUT
        ------
          1. X_f: (4D array) set of filtered EEG signals, shape (trials, channels, time_samples, frequency_bands)
        """
        X_f = np.zeros(
            (X.shape[0], X.shape[1], X.shape[2], self.f_bank.shape[0])
        )  # epochs, Ch, Time, bands
        for f in np.arange(self.f_bank.shape[0]):
            X_f[:, :, :, f] = butterworth_digital_filter(
                X, N=5, Wn=self.f_bank[f], btype='bandpass', fs=self.sfreq
            )
        return X_f

    # ------------------------------------------------------------------------------
    def _sliding_windows(self, X):
        """
        Sliding Windows Characterization.
        INPUT
        -----
          1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
        OUTPUT
        ------
          1. X_w: (4D array) shape (trials, channels, window_time_samples, number_of_windows)
        """
        window_lenght = int(
            self.sfreq * self.vwt[0, 1] - self.sfreq * self.vwt[0, 0]
        )
        X_w = np.zeros(
            (X.shape[0], X.shape[1], window_lenght, self.vwt.shape[0])
        )
        for w in np.arange(self.vwt.shape[0]):
            X_w[:, :, :, w] = X[
                :,
                :,
                int(self.sfreq * self.vwt[w, 0]) : int(
                    self.sfreq * self.vwt[w, 1]
                ),
            ]
        return X_w

    # ------------------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        fit.
        INPUT
        -----
          1. X: (3D array) set of EEG signals, shape (trials, channels, time_samples)
          2. y: (1D array) target labels. Default=None
        OUTPUT
        ------
          1. None
        """
        pass

    # ------------------------------------------------------------------------------
    def transform(self, X, y=None):
        """
        Time frequency representation of EEG signals.
        INPUT
        -----
          1. X: (3D array) set of EEG signals, shape (trials, channels, times)
        OUTPUT
        ------
          1. X_wf: (5D array) Time-frequency representation of EEG signals, shape (trials, channels, window_time_samples, number_of_windows, frequency_bands)
        """
        self._validation_param()  # Validate sfreq, f_freq, vwt

        # Avoid edge effects of digital filter, 1st:fbk, 2th:vwt
        if self.flag_f_bank:
            X_f = self._filter_bank(X)
        else:
            X_f = X[:, :, :, np.newaxis]

        if self.flag_vwt:
            X_wf = []
            for f in range(X_f.shape[3]):
                X_wf.append(self._sliding_windows(X_f[:, :, :, f]))
            X_wf = np.stack(X_wf, axis=-1)
        else:
            X_wf = X_f[:, :, :, np.newaxis, :]
        return X_wf
