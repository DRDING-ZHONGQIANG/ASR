from sphfile import SPHFile
import soundfile as sf
import numpy as np
import glob
import os
import re
import random
from allosaurus.app import read_recognizer
import speech_recognition as sr
import torchmetrics

wer_calc = torchmetrics.WordErrorRate()
r = sr.Recognizer()

timit_prefix = '/home/test/datasets/timit/timit/train'  # prefix for all timit files
num_files_to_test = 20  # how many text files are used in the test
SNRs_dB = [15, 10, 5, 0, -5]
SNR_WERs = []

# ----------------------------------------------------------------
# %%
# Calculate random noise with zero mean at given dB relative to 0
def makeNoise(noise_power_dB, noise_len):
    # Calculate noise and convert it to watts
    noise_pwr = 10 ** (noise_power_dB / 10)
    noise_mean = 0
    noise_signal = np.random.normal(noise_mean, np.sqrt(noise_pwr), noise_len)
    return noise_signal


def signalPower(x):
    return np.average(x ** 2)


def signalPowerdB(x):
    return 10 * np.log10(signalPower(x))


def SNR(noisy_signal, noise):
    powS = signalPower(noisy_signal)
    powN = signalPower(noise)
    return 10.0 * np.log10((powS - powN) / powN)


# %%
# ----------------------------------------------------------------


# Scan all the wav files in the test set
path_sph = timit_prefix + '/*/*/*.wav'
# Compute their full path and other information
sph_files = glob.glob(path_sph)
# How many timit wav files do we have?
num_utterances = len(sph_files)
print(num_utterances, " utterances")
# decide which timit files we will do the testing with, set a random seed so we always get the same set
random.seed(0)
files_to_test = random.sample(sph_files, num_files_to_test)

# Try this at several noise levels
for targetSNR in SNRs_dB:

    wer_array = []
    # Scan through all the files to test
    for wav_file in files_to_test:
        # sph = SPHFile(wav_file)
        # print('Read from ', sph.filename)
        audio, fs = sf.read(wav_file)
        ch = len(audio.shape)
        print('Sample rate=', fs, ' Length=', len(audio))
        # fs = sph.format['sample_rate']
        # ch = sph.format['channel_count']
        # do a basic check
        if fs + ch != 16001:
            print('Skip file because either sample rate or channels is wrong')
            continue
        # read the file
        audio_samples = audio #sph.content
        audio = np.float32(audio_samples)

        # get the phones too
        fn_name = os.path.splitext(wav_file)
        phn_file = fn_name[0] + '.phn'
        txt_file = fn_name[0] + '.txt'

        # print('Reading phone file ',phn_file)
        # print('Reading text file ',txt_file)

        with open(phn_file, 'r') as f:
            phones = [line.rstrip() for line in f]

        # %%
        f = open(txt_file, 'r')
        text_line = f.read()
        f.close()
        # Remove the timing information from the start
        text = text_line.split()[2:]
        # Now combine to one sentence
        sentence = ''
        for i in text:
            sentence += i + ' '
        # Tidy to all lower case and remove punctuation and any trailing/leading whitespace
        sentence = sentence.lower()
        sentence = sentence.strip()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # number of words in the sentence
        len_sentence = len(re.findall(r'\w+', sentence))

        # print('Phone content=',phones)
        # print('Text content=',sentence)

        # %%
        # Now convert the timit phones to a list
        phnphon = []
        for i in phones:
            phnphon.append(i.split()[2])

        # %%
        # We need to add noise to the files
        # Normalise the audio
        audio = (audio - np.average(audio)) / np.max(np.abs(audio))
        # Start looking at the powers
        speechdB = signalPowerdB(audio)
        noisedB = speechdB - targetSNR
        la = len(audio)
        nn = makeNoise(noisedB, la)

        noisy_speech = audio + nn
        actualSNR = SNR(noisy_speech, nn)

        print('Target SNR level (dB)=', targetSNR)
        print('Measured SNR level (dB)=', actualSNR)

        # %%
        # Do the ASR (can easily switch from google to alternatives)
        audio_max = noisy_speech * 32767 / max(abs(noisy_speech))
        audio_max_i = np.int16(audio_max)
        recsent = r.recognize_sphinx(sr.AudioData(audio_max_i, fs, 2), language='en-US')
        #recsent=r.recognize_google(sr.AudioData(audio_max_i,fs,2))
        # Tidy to all lower case and remove punctuation and any trailing/leading whitespace
        recsent = recsent.lower()
        recsent = recsent.strip()
        recsent = re.sub(r'[^\w\s]', '', recsent)
        # number of words in the recognised sentence
        len_recsent = len(re.findall(r'\w+', recsent))

        # %%
        # Somewords are hyphenated or spaced which could be together
        # Simple way to flag is to detect different number of words in a sentence
        # Then we need to do something about that
        if len_recsent != len_sentence:
            words_recsent = recsent.split()
            words_sentence = sentence.split()
            len_min = np.min([len_recsent, len_sentence])
            for i in range(len_min - 1):
                if (words_recsent[i] != words_sentence[i]):
                    # print('Words missing:',words_recsent[i],words_sentence[i])
                    # does it make sense if we add the next word
                    if (words_recsent[i] == words_sentence[i] + words_sentence[i + 1]):
                        # print('Got it')
                        words_sentence[i] = words_sentence[i] + words_sentence[i + 1]
                        words_sentence[i + 1:] = words_sentence[i + 2:]
            # Now piece back the corrected sentences
            orig_sentence = sentence
            sentence = ''
            for i in words_sentence:
                sentence += i + ' '
            sentence = sentence.strip()
            # print('Original sentence: ',orig_sentence)
            # print('Corrected sentence: ',sentence)

        # %%
        # Calculate WER
        wer = wer_calc(sentence, recsent).numpy()
        # print('WER=',wer,'between:\n\tTXT=',sentence,'\n\tASR=',recsent)

        # Only print detailed info if wer is non-zero
        if (wer != 0):
            print('\nWER in ', wav_file)
            print('WER=', wer, 'between:\n\tTXT=', sentence, '\n\tASR=', recsent)
            if len_recsent != len_sentence:
                print('The sentence was corrected')
                print('\toriginal=', orig_sentence)
                print('\tcorrected=', sentence)

        if (wer == np.inf):
            continue  # can't use this result as it is meaningless!

        # accumulate the wer statistics
        wer_array.append(wer)

    # %%
    # Print the final output
    print('=============================================')
    print('Statistics at noise SNR:', actualSNR)
    print('Files tested=', len(files_to_test))
    print('WER values=', len(wer_array))
    print('Average WER=', np.average(wer_array))
    SNR_WERs.append(np.average(wer_array))

# %%
print('=============================================')
print('Final statistics')
for i in range(len(SNRs_dB)):
    print('WER=', SNR_WERs[i], ' at SNR ', SNRs_dB[i], 'dB')