from __future__ import division
from scipy.signal import correlate
from scipy.io.wavfile import read
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

class Spectrum(object):
    """Takes multiple audio clips, converts to frequency spectrum, and compares likness of files

    Followed by longer description

    Attributes: audio signals
                sample frequency
                number of samples
                magnitude of frequencies
                frequencies in hertz
                raw sound data
                raw fourier data 
    Functions:  creates frequency spectrum
                plots raw audio signal
                plots frequency spectrum
                compares audio signals
                prints correlations

    """
    def __init__(self, audio_file, name, data_type=16):
        self.audio_file = audio_file
        self.name = name
        self.data_type = data_type
        self.create_spectrum()
        

    def create_spectrum(self):
        self.sample_frequency, data = read(self.audio_file)
        self.sound = data / (2.0 ** (self.data_type - 1))
        # Find song sample parameters
        if self.sound.ndim==1:
            n_channels =1
        else:
            n_channels = self.sound.shape[1]
            
        self.signal = np.fft.rfft(self.sound)
        if self.signal.ndim>1:
            self.signal=np.mean(self.signal,axis=1)  # average both channels
        
        self.n_samples=self.signal.size
        
        self.magnitude = np.abs(self.signal)/self.n_samples
        self.magnitude = np.asarray(self.magnitude)
        f = [(j*1.0/self.n_samples)*self.sample_frequency 
             for j in range(self.n_samples//2+1)]
        self.frequencies = np.asarray(f)


    def plot_audio(self, ax):
        duration = self.n_samples/ self.sample_frequency
        t = np.arange(0, duration, 1/self.sample_frequency)
        ax.plot(t,self.sound)
        ax.set_title('Graphing sound {}'.format(self.name))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

    def plot_spectrum(self, ax):
         #starting and ending frequencies for
        freq_range=[200,10000]
        index_start=(freq_range[0]*self.n_samples)/self.sample_frequency
        index_end=(freq_range[1]*self.n_samples)/self.sample_frequency

        ax.plot(self.frequencies[index_start:index_end],
                self.magnitude[index_start:index_end])
        ax.set_xlim([freq_range[0],freq_range[1]])
        ax.set_title('Fouier Transform {}'.format(self.name))
        ax.set_xlabel('Frequencies (Hz)')
        ax.set_ylabel('Intensity')

    def compare_with_others(self, other_spectra):
        corr=np.zeros(len(other_spectra))
        freq_range=[200,10000]
        index_start=(freq_range[0]*self.n_samples)/self.sample_frequency
        index_end=(freq_range[1]*self.n_samples)/self.sample_frequency

        for i, other_spectrum in enumerate(other_spectra):
            corr[i] = np.corrcoef(self.magnitude[index_start:index_end], 
                                  other_spectrum.magnitude[index_start:index_end])[0,1]
    #         print('{} and {} are correlated as {}'.format(filenames[i],filenames[j],corr[i,j]))
            if corr[i]>0.3:
                print('{} and {} are the same voice'.format(self.name,other_spectrum.name))
            else:
                print('{} and {} are not the same voice'.format(self.name,other_spectrum.name))

    def plot_correlation(self, other_spectra):
        # Print correlations
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        corr = np.zeros((len(filenames), len(filenames)))
        for i in range(len(filenames)):
            for j in range(len(filenames)):
                corr[i,j] = np.corrcoef(magnitude[i], magnitude[j])[0,1]
        h = ax.matshow(corr, vmin=-1, vmax=1)
        cbar = fig.colorbar(h)

def compare_spectrua(all_spectra):
    pass

if __name__ == '__main__':
    filename = '/home/jeremy/ipython_notebook/spectrum_project/jeremy1.wav'
    first_spectrum = Spectrum(filename,'Jeremy1')
    filename2 = '/home/jeremy/ipython_notebook/spectrum_project/jeremy2.wav'
    second_spectrum = Spectrum(filename2,'Jeremy2')

    all_spectra = [first_spectrum,second_spectrum]

    fig,ax_array=plt.subplots(len(all_spectra),2,figsize=(12,10))
    print(ax_array)
    for i, spectrum in enumerate(all_spectra):
        spectrum.plot_audio(ax_array[i][0])
        spectrum.plot_spectrum(ax_array[i][1])
    compare = spectrum.compare_with_others(all_spectra)

    plt.show()
    print(spectrum.frequencies)