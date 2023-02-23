#Main functions for plotting 

#global imports 
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import plotly.graph_objects as go
import plotly.express as px
from collections import OrderedDict
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

from generalfunc import col_to_string

#local imports
from standardizer import read_phn,TIMIT_to_IPA,IPA_to_TIMIT
from phone_error_rate import error_rate, load_asr_dict,compare_phonemes_perc,compare_phn_wrd_noise_multi
from timit_load import TIMIT_file

from allosaurus.app import read_recognizer
import speech_recognition as sr


def speech_recog(timit_wav):
    r = sr.Recognizer()
    with sr.AudioFile(timit_wav) as source:
        audio = r.record(source)

    return r.recognize_google(audio)

def allosaurus_model(file_directory, fr=16000, dataframe=False):
    model = read_recognizer()
    str_output = model.recognize(file_directory, lang_id='eng', timestamp=True)
    lst_output = str_output.split("\n")
    df = pd.DataFrame(lst_output, columns=['header'])
    df = df.header.str.split(pat=' ', expand=True)
    df.columns = ['start', 'timing', 'phoneme']

    # edit dataframe (add 'timing' to 'start' to get 'end' time/change start & end to milliseconds)
    df['start'] = df['start'].astype(float)
    df['start'] = df['start'].values * fr

    df['timing'] = df['timing'].astype(float)
    df['timing'] = df['timing'].values * fr

    df['end'] = df.apply(lambda row: row.start + row.timing, axis=1)
    finaldf = df[['start', 'end', 'phoneme']]

    if dataframe == True:
        return finaldf
    else:
        allosaurus_phn = col_to_string(finaldf, colname='phoneme')
    return allosaurus_phn


def phn_boxplot(phn_counter_dict, styling_outliers = 'suspectedoutliers'):

  ### <Purpose of function>: Plot boxplot
  ### <Input variables>    : phn_counter_dict = Dictionary containing phoneme as key and list of % correctly predicted phonemes as value.
  ###                        styling_outliers = All Points - "all" / 
  ###                                           Only Whiskers - False / 
  ###                                           Suspected Outliers - suspectedoutliers /
  ###                                           Whiskers and Outliers - "outliers"
  ###                                           (Default set to "Suspected Outliers", full detail: https://plotly.com/python/box-plots/)
  ### <Output>             : Boxplot

  #Initiate
  fig = go.Figure()
  
  #Maintain order of dictionary as written 
  ordered_dict = OrderedDict(phn_counter_dict.items())

  for key in ordered_dict:
    fig.add_trace(go.Box(
        y=ordered_dict[key],
        name=key,
        boxpoints= styling_outliers,
        marker=dict(
            line=dict(
                outliercolor='rgba(219, 64, 82, 0.6)'
    ))))

  #x-axis
  fig.update_layout(title_text=f"Phoneme Accuracy Rate")
  fig.update_xaxes(
          tickangle = 90,
          title_text = "Phonemes",
          title_standoff = 25)
  
  #y-axis
  fig.update_yaxes(
          title_text = "Accuracy (%)",
          title_standoff = 25)

  fig.show()

def noise_stacked_boxplot(error_rate_df,
                          x_axis = "Error Rate (%)",
                          y_axis = "Volume",
                          type_error = "Type of Error"):
  
  ### <Purpose of Function> : Plot stacked boxplot from y-axis
  ### <Input variables>     : error_rate_df = dataframe with Columns ("Volume", "Error Rate (%)", "Type of Error")
  ### <Output>              : Stacked boxplot

  fig = px.bar(error_rate_df, x = x_axis, y = y_axis, color = type_error, barmode = 'stack',height=800)
  fig.show()

### SIMPLIFIED FUNCTIONS (MAIN FUNCTIONS)
#simplified function to plot phn boxplot 
def full_phn_boxplot(asr_model,
                     TIMIT_dict,
                     file_set="TRAIN",
                     DR=[0,None],
                     styling_outliers = False):
    asr_dict = load_asr_dict(TIMIT_dict=TIMIT_dict,asr_model=asr_model,DR=DR,file_set=file_set)
    phn_counter_dict = compare_phonemes_perc(TIMIT_dict=TIMIT_dict,asr_dict=asr_dict,DR=DR,file_set=file_set)

    return phn_boxplot(phn_counter_dict,styling_outliers=styling_outliers)

def full_noise_stackedplot(audio_dict,
                            noise_wav,
                            asr_phn_model,
                            asr_txt_model,
                            cfg_filedir= 'noisyspeech.cfg',
                            DR = [0,None],
                            SPK = [0,None],
                            louder_volumes=[],
                            softer_volumes=[]):

    #Full description of inputs is in utils.phone error_rate func compare_phn_wrd_noise_multi
    error_rate_df = compare_phn_wrd_noise_multi(audio_dict = audio_dict,
                                                noise_wav = noise_wav,
                                                cfg_filedir = cfg_filedir,
                                                asr_phn_model = asr_phn_model,
                                                asr_txt_model = asr_txt_model,
                                                DR = DR,
                                                SPK = SPK,
                                                louder_volumes = louder_volumes,
                                                softer_volumes = softer_volumes)


    error_rate_df.to_csv('PER_WER.csv')

    return noise_stacked_boxplot(error_rate_df)

#Function to plot wav graph showing phoneme per time and error occur at which frames
def phoneme_wavchart(timit_phndir, timit_wavdir,asr_model,vlinecolor='grey',print_df=False):

  phn_file = timit_phndir
  samplerate, data = read(timit_wavdir)

  time = np.arange(0,len(data))

  #initiate plot
  plt.figure(figsize=(20,10))
  plt.plot(time,data)

  #set up vlines
  list_timing = read_phn(phn_file,df=True)['end'].tolist()
  list_phn    = read_phn(phn_file,df=True)['phoneme'].tolist()
  list_timing = [int(i) for i in list_timing]
  plt.vlines(x=list_timing,ymin=-35000,ymax=35000,colors = vlinecolor,linestyle='dotted')
  for i,timing in enumerate(list_timing):
    plt.text(timing,-36000,timing,rotation=90,horizontalalignment='right')
    if i%2 == 0:
      plt.text(timing,35000,list_phn[i],verticalalignment = 'top',horizontalalignment='center', fontsize='x-large')
    else:
      plt.text(timing,33000,list_phn[i],verticalalignment = 'top',horizontalalignment='center', fontsize='x-large')

  ### plot highlights of wrong area
  timittest = TIMIT_to_IPA(read_phn(phn_file,string=True))[1:-1]     #TIMIT phn #[1:-1] to remove the '/' 
  asrtest = IPA_to_TIMIT(asr_model(timit_wavdir,dataframe=False)) #ASR phn

  error_rate_df,tracker_df = error_rate(timittest,asrtest)           #get tracker_df
  tracker_df.index+=1                                                #shift index of tracker_df up by 1

  timit_phn = read_phn(phn_file,df=True).iloc[1:-1]                  #timit phn (dataframe)
  merged_df = pd.concat([timit_phn,tracker_df],axis=1)               #concat left df

  merged_df['start'] = merged_df['start'].astype(int)                #convert string to int
  merged_df['end'] = merged_df['end'].astype(int)
  merged_df.columns = ['start','end','phoneme','initial_phoneme','error','substituted'] # rename col name

  #select for substitution errors
  sub_df = merged_df[merged_df['error'] == 'Substitution'][['start','end']]
  for i in range(len(sub_df)):
    if i==0:
      plt.axvspan(sub_df.iloc[i][0], sub_df.iloc[i][1], color='orange', alpha=0.4,label='substituted')
    else:
      plt.axvspan(sub_df.iloc[i][0], sub_df.iloc[i][1], color='orange', alpha=0.4)

  #select for deletion errors
  del_df = merged_df[merged_df['error'] == 'Deletion'][['start','end']]
  for i in range(len(del_df)):
    if i==0:
      plt.axvspan(del_df.iloc[i][0], del_df.iloc[i][1], color='red', alpha=0.4,label='deleted')
    else:
      plt.axvspan(del_df.iloc[i][0], del_df.iloc[i][1], color='red', alpha=0.4)

  #labels
  plt.xlabel(f'Time [{samplerate} samples/s]')
  plt.ylabel('Amplitude') 
  plt.title('Plot showing phonemes per frame')
  plt.legend(bbox_to_anchor=(1.05, 1),loc='center')
  plt.show()
  
  if print_df == True:
    print("Dataframe Showing Substitution")
    print(merged_df[merged_df['error'] == 'Substitution'][['start','end','phoneme','substituted']])

    print("Dataframe Showing Deletion")
    print(merged_df[merged_df['error'] == 'Deletion'][['start','end','phoneme']])


#-----------------------------------------------------------
timit_dict   = TIMIT_file('/home/test/datasets/timit/timit')
#phn_file_dir = timit_dict['train']['dr1']['fecd0']['phn'][0]

from allosaurus.app import read_recognizer
asr_model = read_recognizer().recognize

#full_phn_boxplot(allosaurus_model,  timit_dict,   file_set="train",  DR=[0,None],  styling_outliers = False)

phn_file_dir = timit_dict['train']['dr1']['fecd0']['phn'][0]; wav_file_dir = timit_dict['train']['dr1']['fecd0']['wav'][0]

phoneme_wavchart(timit_phndir = phn_file_dir,  timit_wavdir = wav_file_dir,  asr_model=allosaurus_model, vlinecolor='grey',  print_df=False)

'''
full_noise_stackedplot(audio_dict=timit_dict['train'],
                               noise_wav="audiocheck.net_whitenoisegaussian.wav",
                               cfg_filedir= 'noisyspeech.cfg',
                               asr_phn_model=allosaurus_model,
                               asr_txt_model=speech_recog,
                               DR = [0,1],
                               SPK = [0,1],
                               louder_volumes=[],
                               #softer_volumes=[5,10,15,20,25,30])
                               softer_volumes = [10,5,0,-5,-10])
'''








