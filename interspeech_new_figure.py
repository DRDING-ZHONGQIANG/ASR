import numpy
import difflib
import pandas
import sns as sns
from sphfile import SPHFile
import soundfile as sf
import glob
import os
import random
from allosaurus.app import read_recognizer
import speech_recognition as sr

#----- timit_to_ipa ______ allophones_to_ipa

timit_phns = ['h#','ax', 'ax-h', 'axr', 'b', 'bcl', 'd','dcl','dx','eng','epi','g','gcl','hv','h#','ix','kcl','k','nx','p','pau','pcl','t','tcl','ux', \
               'aa', 'ae', 'ah', 'ah0', 'ao','aw', 'ay','eh','er', 'er0', 'ey','ih', 'ih0', 'iy', 'ow', 'oy', 'uh', 'uw', 'b', 'ch', 'd', 'dh',\
               'el', 'em', 'en', 'f', 'g', 'hh', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'q', 'r', 's', 'sh', 't', 'th','v','w','wh','y','z','zh','ɹ̩']
timit_ipa  = ['', 'ə','ə̥', 'ɚ','b','','d','','ɾ','ŋ̍','','g','','h','','ɨ','','k','ɾ̃','','p','','t','','ʉ', 'ɑ','æ','ʌ','ə','ɔ','aʊ','aɪ','ɛ','ɝ','ɚ','eɪ',\
              'ɪ','ɨ','i','oʊ','ɔɪ','ʊ','u','b','tʃ','d','ð', 'l ', 'm̩', 'n̩', 'f','ɡ', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'ŋ', 'p', 'ʔ', 'ɹ', 's',\
              'ʃ', 't', 'θ', 'v', 'w', 'ʍ', 'j', 'z', 'ʒ','ɹ']

allo_phn   = ['pʰ','tʰ','kʰ', 'b̥','d̥', 'ɡ̥', 'a', 'ɹ̩', 'r','o','e','l̪','ɒ', 'ʁ','ø','uː','iː', 'x', 'ɯ', 'ɻ']
allo_ipa   = ['p','t','k','b','d','g','ɑ', 'ɹ', 'ɹ','oʊ','ɛ','l','ɑ','ɹ', 'θ','ʉ','ɨ', 's', 'w', 'ɹ']

#Scanlon, P.; Ellis, D. & Reilly, R. (2007). Using Broad Phonetic Group Experts for Improved Speech Recognition.
# IEEE Transactions on Audio, Speech and Language Processing,vol.15 (3) , pp 803-812, March 2007, ISSN 1558-7916.
timit_vowels     = ['aa', 'ae', 'ah', 'ah0','ao', 'ax', 'ax-h', 'axr', 'ay', 'aw', 'eh', 'el', 'er', 'er0','ey', 'ih', 'ih0', 'ix', 'iy', 'l', 'ow', 'oy', 'r', 'uh', 'uw', 'ux', 'w', 'y']
timit_stops      = ['p', 't', 'k', 'b', 'd', 'g', 'jh', 'ch']
timit_fricatives = ['s', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh', 'hv']
timit_nasals      = ['m', 'em', 'n', 'nx', 'ng', 'eng', 'en']

# 0 - vowels; 1 - stops; 2 - fricatives; 3 - nasals; 4 - silences
phone_class_vowels     = 0
phone_class_stops      = 1
phone_class_fricatives = 2
phone_class_nasals     = 3
phone_class_silences   = 4

def timit_phone_classification(letter):
    pos      = timit_ipa.index(letter)
    timit_label = timit_phns[pos]
    if timit_label in timit_vowels:
        return phone_class_vowels
    if timit_label in timit_stops:
        return phone_class_stops
    if timit_label in timit_fricatives:
        return phone_class_fricatives
    if timit_label in timit_nasals:
        return phone_class_nasals
    else:
        return phone_class_silences

def allo_to_ipa(phones):
    plist = phones.split(' ')
    for i in range(len(plist)):
        if plist[i] in allo_phn:
            pos = allo_phn.index(plist[i])
            if pos >=0:
                plist[i] =  allo_ipa[pos]
    return plist

def timit_to_ipa(phones):
    plist = phones.split('/')
    for i in range(len(plist)):
        if plist[i] in timit_phns:
            pos = timit_phns.index(plist[i])
            if pos >=0:
                plist[i] =  timit_ipa[pos]
    return plist
# ************************ main program start here *********************
# global variables: target_phone_class_set, predicted_phone_class_set

target_phone_class_set    = []
predicted_phone_class_set = []
target_phone_list         = []
predicted_phone_list      = []

timit_prefix = '/home/test/datasets/timit/timit/train'  # prefix for all timit files
num_files_to_test = 100  # how many text files are used in the test
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
for wav_file in files_to_test:
    #sph = SPHFile(wav_file)
    #print('Read from ', sph.filename)
    audio, fs = sf.read(wav_file)
    ch = len(audio.shape)
    print('Sample rate=',fs,' Length=',len(audio))
    #fs = sph.format['sample_rate']
    #ch = sph.format['channel_count']
    # do a basic check
    if fs + ch != 16001:
        print('Skip file')
        continue
    # read the file
    #audio = sph.content
    # get the phones too
    fn_name = os.path.splitext(wav_file)
    phn_file = fn_name[0] + '.phn'

    txt_file = fn_name[0] + '.txt'
    print('Reading phone file ', phn_file)
    print('Reading text file ', txt_file)

    with open(phn_file, 'r') as f:
        phones = [line.rstrip() for line in f]

    f = open(txt_file, 'r')
    text = f.read()
    f.close()

    print('Phone content=', phones)
    print('Text content=', text)
    # Now convert the timit phones to a list
    # for i in phones:
    #    phnphon = i.split()[2]

    phn_df = pandas.read_csv(phn_file, names=['header'])
    phn_df = phn_df.header.str.split(pat=' ', expand=True)
    phn_df.columns = ['start', 'end', 'phoneme']
    phn_string = '/'.join(phn_df['phoneme'].tolist())
    target_phn = timit_to_ipa(phn_string)
    target_phn = [x for x in target_phn if x != '']

    asr_model = read_recognizer('latest')
    pred_phn = asr_model.recognize(wav_file,'eng')
    pred_phn = allo_to_ipa(pred_phn)

    seq_matcher = difflib.SequenceMatcher(None, target_phn, pred_phn)
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "equal":
              for letter in target_phn[i1:i2]:  # dealing with list of multiple correct phonemes
                  phone_class = timit_phone_classification(letter)
                  target_phone_class_set.append(str(phone_class))
                  predicted_phone_class_set.append(str(phone_class))
                  target_phone_list.append(letter)
                  predicted_phone_list.append(letter)

        if tag == 'replace':
              offset = 0
              for letter in target_phn[i1:i2]:
                  target_phone_class = timit_phone_classification(letter)
                  target_phone_class_set.append(str(target_phone_class))
                  if j1 + offset < j2:
                      predicted_letter = pred_phn[j1+offset]
                  else:
                      predicted_letter = pred_phn[j2-1]
                  predicted_phone_class = timit_phone_classification(predicted_letter)
                  predicted_phone_class_set.append(str(predicted_phone_class))
                  target_phone_list.append(letter)
                  predicted_phone_list.append(predicted_letter)
                  offset += 1

        if tag == 'delete':
              for letter in target_phn[i1:i2]:
                 phone_class = timit_phone_classification(letter)
                 target_phone_class_set.append(str(phone_class))
                 predicted_phone_class_set.append(str(phone_class_silences))
                 target_phone_list.append(letter)
                 predicted_phone_list.append('_')

numpy.save("target_phone_class",target_phone_class_set)
numpy.save("predicted_phone_class", predicted_phone_class_set)
numpy.save("predicted_phones", predicted_phone_list)
numpy.save("target_phones", target_phone_list)

phone = []
phone_total   = []
phone_correct = []
for i in range(len(target_phone_list)):
    if target_phone_list[i] not in phone:
        phone.append(target_phone_list[i])
        phone_total.append(int(1))
        if target_phone_list[i]==predicted_phone_list[i]:
            phone_correct.append(int(1))
        else:
            phone_correct.append(int(0))
    else:
        inx = phone.index(target_phone_list[i])
        phone_total[inx] += 1
        if target_phone_list[i] == predicted_phone_list[i]:
            phone_correct[inx] += 1

final_symbol  = []
final_score   = []
final_correct = []
final_total   = []
for i in range(len(phone)):
    if phone_correct[i] > 0:
       final_symbol.append(phone[i])
       final_score.append(float(1-phone_correct[i]/phone_total[i]))
       final_correct.append(phone_correct[i])
       final_total.append(phone_total[i])

for i in range(len(final_symbol)):
    print(final_symbol[i], final_score[i], final_correct[i], final_total[i])

#---------------------------------- plot confusion matrix-------------------------
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


y_true = np.load('target_phone_class.npy')
y_pred = np.load('predicted_phone_class.npy')
fig, ax = plt.subplots(1, 2, figsize=(15,5))
sns.set()


y_true = list(map(int, y_true))
y_pred = list(map(int, y_pred))
C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4], normalize="true")
print(C2)
plt.subplot(121)
sns.heatmap(C2, annot=True, ax=ax[0], cmap="crest", vmin=0.2, vmax=0.4)


scale = np.arange(5)
# index = ["Vowels","Stops","Fricatives","Nasals","Silences"]
index = ["VW","ST","F","NS","SL"]
ax[0].set_title('confusion matrix')
plt.sca(ax[0])
plt.xticks(scale+0.5, index, rotation=0)
plt.yticks(scale+0.5, index, rotation=90)
ax[0].set_xlabel('Prediction')
ax[0].set_ylabel('Ground Truth')

# PER = [0.027,0.038,0.048,0.051,0.087,0.095,0.106,0.115]
# phone = ['t','k','ɹ','m','l','s','n','ɑ']
#PER = [0.986,0.985,0.938,0.846,0.615,0.500]
#phone= ['ʔ','ɨ','ʉ','p','ʊ','ʒ']
PER = [0.027,0.038,0.048,0.938,0.985,0.986]
phone = ['t','k','ɹ','ʉ','ɨ','ʔ']
#IVM normalise PER
#PER=PER/np.max(PER)

xticks=np.arange(0,max(PER),0.2)
plt.subplot(122)
sns.barplot(y=phone, x=PER)
# ax[1].set_title('Least Confused Phones')
#plt.xticks(xticks)
ax[1].set_title('Most and least confused phones')
ax[1].set_xlabel('relative PER')

plt.savefig('savedplot.pdf', dpi=300)
plt.show()
