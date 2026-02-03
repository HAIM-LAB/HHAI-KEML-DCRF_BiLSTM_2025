import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import os, sys, requests, csv
#from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import librosa 
import librosa.display
from IPython.display import Audio
import glob

ravdess_path = '..\\workspace\\2025_ACL_SpeechEMD_proj_code\\dataset\\Audio_Speech_Actors_01-24\\'
tess_path = '..\\workspace\\2025_ACL_SpeechEMD_proj_code\\dataset\\TESS\\'
savee_path = '..\\workspace\\2025_ACL_SpeechEMD_proj_code\\dataset\\SAVEE\\'
cremaD_path = '..\\workspace\\2025_ACL_SpeechEMD_proj_code\\dataset\\CREMA-D\\'
emodb_path = '..\\workspace\\2025_ACL_SpeechEMD_proj_code\\dataset\\EmoDB\\'

# Reference code from https://www.kaggle.com/code/preethikurra/emotion-recognition
def read_ravdess_data():
    
    ravdess_directory_list = os.listdir(ravdess_path)
    #ravdess_directory_list.remove('.DS_Store')
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor = os.listdir(ravdess_path + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            emotion = int(part[2]) - 1; #-1 so that to start from 0
            if emotion == 1: #avoid calm
                continue;
            elif emotion > 1:
                emotion = emotion - 1
            #file_emotion.append(int(part[2]))
            file_emotion.append(emotion)
            file_path.append(ravdess_path + dir + '/' + file)
        
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    # changing integers to actual emotions.
    #Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    Ravdess_df.Emotions.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}, inplace=True)
    Ravdess_df.head()
    return file_path, file_emotion, Ravdess_df

def read_tess_data():
    
    tess_directory_list = os.listdir(tess_path)
    # Define the emotion mapping dictionary
    emotion_map = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'fear',
        5: 'disgust',
        6: 'surprise'
    }

    # Inverse mapping for quick lookup of integer from string
    emotion_map_reverse = {v: k for k, v in emotion_map.items()}

    file_emotion = []
    file_path = []
    for file in tess_directory_list:
        part = file.split('.')[0]
        part = part.split('_')[2]
        '''
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        '''
        # Check if the emotion part is 'ps' for 'surprise'
        if part == 'ps':
            # Append the integer corresponding to 'surprise'
            file_emotion.append(emotion_map_reverse['surprise'])
        else:
            # Otherwise, append the corresponding integer for the other emotions
            emotion = emotion_map_reverse.get(part, None);
            if emotion >= 0 and emotion <= 6:
                file_emotion.append(emotion)  # Default to None if part is not found in the mapping            
            else:
                continue
    
        file_path.append(tess_path + '/' + file)
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    Tess_df.Emotions.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}, inplace=True)    
    Tess_df.head()
    return file_path, file_emotion, Tess_df

#{0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}
def savee_convert(name):
    ch = name[0]
    if ch == 's':
        ch+=name[1]
    if ch == 'a':
        return 3
    elif ch == 'd':
        return 5
    elif ch == 'f':
        return 4
    elif ch == 'h':
        return 1
    elif ch == 'n':
        return 0
    elif ch == 'sa':
        return 2
    elif ch == 'su':
        return 6
def read_savee_data():
    

    savee_directory_list = os.listdir(savee_path)
    file_emotion = []
    file_path = []
    for dir in savee_directory_list:
        if dir == 'Info.txt':
            continue
        dicrectory = os.listdir(savee_path + dir)
        for file in dicrectory:
            part = file.split('.')[0]
            emotion = savee_convert(part)
            if emotion >= 0 and emotion <= 6:
                file_emotion.append(emotion)  # Default to None if part is not found in the mapping            
            else:
                continue    
            file_path.append(savee_path + dir + '\\' + file)
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    savee_df = pd.concat([emotion_df, path_df], axis=1)
    savee_df.Emotions.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}, inplace=True)    
    savee_df.head()
    return file_path, file_emotion, savee_df    

def cremaD_convert(name):
    #{0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}
    ch = name.split('_')[2]
    if ch == 'ANG':
        return 3
    elif ch == 'DIS':
        return 5
    elif ch == 'FEA':
        return 4
    elif ch == 'HAP':
        return 1
    elif ch == 'NEU':
        return 0
    elif ch == 'SAD':
        return 2
    else:
        return -1
    
def read_cremaD_data():
    

    cremaD_directory_list = os.listdir(cremaD_path)
    file_emotion = []
    file_path = []
    for file in cremaD_directory_list:
        part = file.split('.')[0]
        emotion = cremaD_convert(part)
        if emotion >= 0 and emotion <= 6:
            file_emotion.append(emotion)  # Default to None if part is not found in the mapping            
        else:
            continue    
        file_path.append(cremaD_path + '/' + file)
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    cremaD_df = pd.concat([emotion_df, path_df], axis=1)
    cremaD_df.Emotions.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}, inplace=True)    
    cremaD_df.head()
    return file_path, file_emotion, cremaD_df    
def emodb_convert(part):
    #{0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}
    ch = part[5]
    if  (ch=='W') :
        return 3 #emotion.append('Angry')
    elif (ch=='E') :
        return 5 #emotion.append('Disgust')    
    elif (ch=='F'):
        return 1 #emotion.append('Happy')
    elif (ch=='L') :
        return -1 #emotion.append('Boredom')
    elif (ch=='T') :
        return 2 #emotion.append('Sadness') 
    elif (ch=='A'):
        return 4 #emotion.append('Fear')
    elif (ch=='N'):
        return 0 #emotion.append('Neutral')
    else:
        return -1

def read_emodb_data():
    

    emodb_directory_list = os.listdir(emodb_path)
    file_emotion = []
    file_path = []
    for file in emodb_directory_list:
        part = file.split('.')[0]
        emotion = emodb_convert(part)
        if emotion >= 0 and emotion <= 6:
            file_emotion.append(emotion)  # Default to None if part is not found in the mapping            
        else:
            continue    
        file_path.append(emodb_path + '/' + file)
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    emodb_df = pd.concat([emotion_df, path_df], axis=1)
    emodb_df.Emotions.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}, inplace=True)    
    emodb_df.head()
    return file_path, file_emotion, emodb_df    

def plot_data_emotions(Ravdess_df, Tess_df):

# creating Dataframe using all the 4 dataframes we created so far.
    #data_path = pd.concat([ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
    data_path = pd.concat([Ravdess_df, Tess_df], axis = 0)
    data_path.to_csv("data_path.csv",index=False)
    data_path.head()
    print(data_path.Emotions.value_counts())
    plt.title('Count of Emotions', size=16)
    sns.countplot(data_path.Emotions)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

'''
include more ploting functions from:
https://www.kaggle.com/code/preethikurra/emotion-recognition
'''

# Define emotion mapping
#{0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'disgust', 6:'surprise'}
emotion_mapping = {
    "Angry": 0,
    "Disgusted": 1,
    "Fearful": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Suprised": 6
}

#https://www.kaggle.com/datasets/uldisvalainis/audio-emotions
def read_combined_dataset():
    # Base directory
    base_dir = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\dataset\\Emotions"

    # Lists to store file paths and corresponding emotions
    file_paths = []
    emotion_labels = []

    # Iterate through each emotion folder
    for emotion, code in emotion_mapping.items():
        folder_path = os.path.join(base_dir, emotion)  # Construct folder path
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))  # Get all WAV files

        # Add file paths and corresponding emotion labels
        file_paths.extend(wav_files)
        emotion_labels.extend([code] * len(wav_files))

    # Print example output
    for i in range(5):  # Print first 5 entries for verification
        print(f"File: {file_paths[i]} -> Emotion: {emotion_labels[i]}")
    return file_paths, emotion_labels

    #Ref: https://www.kaggle.com/datasets/uldisvalainis/audio-emotions/data