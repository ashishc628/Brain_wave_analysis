import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import csv
from scipy import signal, stats
import pandas as pd

# Function to smooth data using a triangular filter
def smoothTriangle(data, degree):
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))
    
    # Handle boundaries
    smoothed = [smoothed[0]] * int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    
    return smoothed

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.header("Brain Wave Analysis")
    st.caption('Enhance your understanding of mental wellness through EEG data analysis.')

    uploaded_file = st.file_uploader("Upload a WAV File", type=["wav"])
    
    if uploaded_file is not None:
        fs, data = waves.read(uploaded_file)

        length_data = np.shape(data)
        length_new = length_data[0] * 0.05
        ld_int = int(length_new)

        data_new = signal.resample(data, ld_int)

        st.subheader("Spectrogram")
        plt.figure('Spectrogram', figsize=(10, 5))  # Adjust the figure size here
        d, f, t, im = plt.specgram(data_new, NFFT=256, Fs=500, noverlap=250)
        plt.ylim(0, 90)
        plt.colorbar(label= "Power/Frequency")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        st.pyplot()

        st.markdown("### Spectrogram Explanation:")
        st.write("The spectrogram provides a visual representation of the power spectral density over time and frequency.")
        st.write("It helps visualize how different frequencies contribute to the signal, useful for identifying patterns.")

        st.subheader("Alpha Power Over Time")
        position_vector = []
        length_f = np.shape(f)
        l_row_f = length_f[0]
        for i in range(0, l_row_f):
            if f[i] >= 7 and f[i] <= 12:
                position_vector.append(i)

        length_d = np.shape(d)
        l_col_d = length_d[1] if len(length_d) > 1 else 0
        AlphaRange = [np.mean(d[position_vector[0]:max(position_vector) + 1, i]) for i in range(l_col_d)]

        plt.figure('AlphaRange', figsize=(10, 5))  # Adjust the figure size here
        y = smoothTriangle(AlphaRange, 100)
        plt.plot(t, y)
        plt.xlabel('Time [s]')
        plt.xlim(0, max(t))
        st.pyplot()

        st.markdown("### Alpha Power Over Time Explanation:")
        st.write("This plot shows the variation in alpha power (8-10 Hz) over time after smoothing.")
        st.write("Alpha waves are commonly associated with relaxed states.")

        st.subheader("Statistical Analysis")
        tg = np.array([4.2552, 14.9426, 23.2801, 36.0951, 45.4738, 59.3751, 72.0337, 85.0831, max(t) + 1])
        length_t = np.shape(t)
        l_row_t = length_t[0]
        eyesclosed = []
        eyesopen = []
        j = 0  # Initial variable to traverse tg
        l = 0  # Initial variable to loop through the "y" data
        for i in range(0, l_row_t):
            if t[i] >= tg[j]:
                if j % 2 == 0:
                    eyesopen.append(np.mean(y[l:i]))
                if j % 2 == 1:
                    eyesclosed.append(np.mean(y[l:i]))
                l = i
                j = j + 1

        plt.figure('DataAnalysis', figsize=(8, 4))  # Adjust the figure size here
        plt.boxplot([eyesopen, eyesclosed], sym='ko', whis=1.5)
        plt.xticks([1, 2], ['Eyes open', 'Eyes closed'], size='small', color='k')
        plt.ylabel('Alpha Power')

        st.pyplot()

        meanopen = np.mean(eyesopen)
        meanclosed = np.mean(eyesclosed)
        sdopen = np.std(eyesopen)
        sdclosed = np.std(eyesclosed)

        st.markdown("### Statistical Analysis Explanation:")
        st.write("The box plot visually compares the distribution of alpha power during eyes open and eyes closed states.")
        st.write("The t-test result indicates statistical significance in the difference.")

        st.write("Mean (Eyes Open):", meanopen)
        st.write("Mean (Eyes Closed):", meanclosed)
        st.write("Standard Deviation (Eyes Open):", sdopen)
        st.write("Standard Deviation (Eyes Closed):", sdclosed)

        result = stats.ttest_ind(eyesopen, eyesclosed, equal_var=False)
        st.write("T-Test Result:")
        st.write("t-Statistic:", result.statistic)
        st.write("p-Value:", result.pvalue)

if __name__ == "__main__":
    main()
