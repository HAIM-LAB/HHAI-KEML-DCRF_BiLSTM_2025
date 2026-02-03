import ReadDatabase  as speechread # Import the audio module
import AudioProcess  as ap # Import the audio module
import FeatureExtract as fet
import sys  # Import sys for exiting the program
import random
import os
import AudioProcess as ap
'''
# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

'''
EXPECTED_FEATURES = 190 #190 #162*3# 94*3 #118

train_file_cmb = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\datasetFeatur\\train.feat"
ravdess_features_file = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\datasetFeature\\ravdess.feat"
tess_features_file = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\datasetFeatur\\tess.feat"
cremaD_features_file = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\datasetFeatur\\cremaD.feat"
savee_features_file = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\datasetFeatur\\savee.feat"
emodb_features_file = "..\\workspace\\2025_ACL_SpeechEMD_proj_code\\datasetFeatur\\emodb.feat"

def process_extractFeat_ravdess_dataset():
    ravdess_filepath, ravdess_emotion, Ravdess_df = speechread.read_ravdess_data()
    #For ravdess data:
    ravdess_offset = None #0.30
    ravdess_duration = None #2.75
    #ravdess databse features Write to the file 
    
    total_files = len(ravdess_emotion)
    with open(ravdess_features_file, "w") as train_f:
        count = 0  # Initialize counter
        for emotion, filepath in zip(ravdess_emotion, ravdess_filepath):
            # Load audio
            wav, sr = ap.loadAudio(filepath, ravdess_offset, ravdess_duration)       
            
            # Extract features (N features per audio file)
            result = fet.extract_features(filepath, wav, sr)  # Assuming 'feat' is a NumPy array or list of N features
            for feat in result:
                # Check feature size
                if len(feat) != EXPECTED_FEATURES:
                    print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")
                    continue #failedsys.exit(1)  # Exit the script with error code 1

                # Format each float to 15 digits after the decimal point
                feat_line = " ".join(f"{x:.15f}" for x in feat)
                # Write feature vector
                train_f.write("1\n")
                train_f.write(feat_line + "\n")
                # Write emotion label (integer converted to string)
                train_f.write(f"{emotion}\n")  # Convert int to string explicitly
            # Increment counter
            count += 1
            # Print message every 200 files
            if count % 100 == 0:
                print(f"Ravdess: Processed {count} out of {total_files} audio files...")
    return 1 #success

def process_extractFeat_tess_dataset():
    tess_filepath, tess_emotion, tess_df = speechread.read_tess_data()
    #For ravdess data:
    tess_offset = None
    tess_duration = None

    total_files = len(tess_emotion)
    with open(tess_features_file, "w") as train_f:
        count = 0  # Initialize counter
        for emotion, filepath in zip(tess_emotion, tess_filepath):
            # Load audio
            wav, sr = ap.loadAudio(filepath, tess_offset, tess_duration)       
            
            # Extract features (N features per audio file)
            result = fet.extract_features(filepath, wav, sr)  # Assuming 'feat' is a NumPy array or list of N features
            for feat in result:            
                # Check feature size
                if len(feat) != EXPECTED_FEATURES:
                    print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")
                    continue #failedsys.exit(1)  # Exit the script with error code 1

                # Format each float to 15 digits after the decimal point
                feat_line = " ".join(f"{x:.15f}" for x in feat)
                # Write feature vector
                train_f.write("1\n")
                train_f.write(feat_line + "\n")                
                # Write emotion label (integer converted to string)
                train_f.write(f"{emotion}\n")  # Convert int to string explicitly
            # Increment counter
            count += 1
            # Print message every 200 files
            if count % 200 == 0:
                print(f"Tess:: Processed {count} out of {total_files} audio files...")
    return 1 #success
def process_extractFeat_cremaD_dataset():
    cremaD_filepath, cremaD_emotion, tess_df = speechread.read_cremaD_data()
    cremaD_offset = None
    cremaD_duration = None
    total_files = len(cremaD_emotion)

    with open(cremaD_features_file, "w") as train_f:
        count = 0  # Initialize counter
        for emotion, filepath in zip(cremaD_emotion, cremaD_filepath):
            # Load audio
            wav, sr = ap.loadAudio(filepath, cremaD_offset, cremaD_duration)       
            
            # Extract features (N features per audio file)
            result = fet.extract_features(filepath, wav, sr)  # Assuming 'feat' is a NumPy array or list of N features
            for feat in result:
                # Check feature size
                if len(feat) != EXPECTED_FEATURES:
                    print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")
                    continue #failedsys.exit(1)  # Exit the script with error code 1

                # Format each float to 15 digits after the decimal point
                feat_line = " ".join(f"{x:.15f}" for x in feat)
                # Write feature vector
                train_f.write("1\n")
                train_f.write(feat_line + "\n")                
                # Write emotion label (integer converted to string)
                train_f.write(f"{emotion}\n")  # Convert int to string explicitly
            # Increment counter
            count += 1
            # Print message every 200 files
            if count % 200 == 0:
                print(f"CremaD:: Processed {count} out of {total_files} audio files...")
    return 1 #success

def process_extractFeat_savee_dataset():
    savee_filepath, savee_emotion, savee_df = speechread.read_savee_data()
    savee_offset = None
    savee_duration = None
    total_files = len(savee_emotion)

    with open(savee_features_file, "w") as train_f:
        count = 0  # Initialize counter
        for emotion, filepath in zip(savee_emotion, savee_filepath):
            # Load audio
            wav, sr = ap.loadAudio(filepath, savee_offset, savee_duration)       
            
            # Extract features (N features per audio file)
            result = fet.extract_features(filepath, wav, sr)  # Assuming 'feat' is a NumPy array or list of N features
            for feat in result:            
                # Check feature size
                if len(feat) != EXPECTED_FEATURES:
                    print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")
                    continue #failedsys.exit(1)  # Exit the script with error code 1

                # Format each float to 15 digits after the decimal point
                feat_line = " ".join(f"{x:.15f}" for x in feat)
                # Write feature vector
                train_f.write("1\n")
                train_f.write(feat_line + "\n")                
                # Write emotion label (integer converted to string)
                train_f.write(f"{emotion}\n")  # Convert int to string explicitly
            # Increment counter
            count += 1
            # Print message every 200 files
            if count % 200 == 0:
                print(f"savee:: Processed {count} out of {total_files} audio files...")
    return 1 #success
def process_extractFeat_emodb_dataset():
    emodb_filepath, emodb_emotion, tess_df = speechread.read_emodb_data()
    emodb_offset = None
    emodb_duration = None
    total_files = len(emodb_emotion)

    
    with open(emodb_features_file, "w") as train_f:
        count = 0  # Initialize counter
        for emotion, filepath in zip(emodb_emotion, emodb_filepath):
            # Load audio
            wav, sr = ap.loadAudio(filepath, emodb_offset, emodb_duration)       
            
            # Extract features (N features per audio file)
            result = fet.extract_features(filepath, wav, sr)  # Assuming 'feat' is a NumPy array or list of N features
            for feat in result:            
                # Check feature size
                if len(feat) != EXPECTED_FEATURES:
                    print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")
                    continue #failedsys.exit(1)  # Exit the script with error code 1

                # Format each float to 15 digits after the decimal point
                feat_line = " ".join(f"{x:.15f}" for x in feat)
                # Write feature vector
                train_f.write("1\n")
                train_f.write(feat_line + "\n")                
                # Write emotion label (integer converted to string)
                train_f.write(f"{emotion}\n")  # Convert int to string explicitly
            # Increment counter
            count += 1
            # Print message every 200 files
            if count % 200 == 0:
                print(f"emodb:: Processed {count} out of {total_files} audio files...")
    return 1 #success

def process_extract_combined_dataset():
    
    cmb_filepath, cmb_emotion = speechread.read_combined_dataset()
    #For ravdess data:
    cmb_offset = None
    cmb_duration = None
    #ravdess databse features Write to the file 
    total_files = len(cmb_emotion)
    #with open(train_file_cmb, "w") as train_f, open(validate_file_cmb, "w") as test_f:
    with open(train_file_cmb, "w") as train_f:
        count = 0  # Initialize counter
        for emotion, filepath in zip(cmb_emotion, cmb_filepath):
        #for emotion, filepath in zip(cmb_emotion[start_idx:end_idx], cmb_filepath[start_idx:end_idx]):
            # Load audio
            wav, sr = ap.loadAudio(filepath, cmb_offset, cmb_duration)       
            #print(filepath)
            # Extract features (N features per audio file)
            result = fet.extract_features(filepath, wav, sr)  # Assuming 'feat' is a NumPy array or list of N features
            for feat in result:            
                # Check feature size
                if len(feat) != EXPECTED_FEATURES:
                    print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")
                    continue #failedsys.exit(1)  # Exit the script with error code 1

                # Format each float to 15 digits after the decimal point
                feat_line = " ".join(f"{x:.15f}" for x in feat)
                # Write feature vector
                train_f.write("1\n")
                train_f.write(feat_line + "\n")                
                # Write emotion label (integer converted to string)
                train_f.write(f"{emotion}\n")  # Convert int to string explicitly
            count += 1
            # Print message every 200 files
            if count % 200 == 0:
                print(f"Tess:: Processed {count} out of {total_files} audio files...")
    return 1 #success

def test_audio():
    #for path in ravdess_filepath:
    out_test_m = "..\\dataset\\test_m.wav"
    out_test_o = "..\\dataset\\test_output.wav"

    tess_offset = 0 #0.30
    tess_duration = None #2.75
    ravdess_offset = 0.30
    ravdess_duration = 2.75

    #wav, sr = ap.loadAudio(ravdess_filepath[226],ravdess_offset,ravdess_duration)
    wavfile = "..\\dataset\\TESS\\OAF_bean_angry.wav"
    #wavfile = "..\\dataset\\Audio_Speech_Actors_01-24\\Actor_20\\03-01-06-01-01-02-20.wav"
    #wavfile = "..\\dataset\\Emotions\\Sad\\1076_MTI_SAD_XX.wav"
    #wav, sr = ap.loadAudio(tess_filepath[226],tess_offset,tess_duration)
    #wavfile = "..\\dataset\\CREMA-D\\1001_IEO_ANG_HI.wav"
    wav, sr = ap.loadAudio(wavfile,ravdess_offset,ravdess_duration)
    ap.outputAudio(out_test_m, wav, sr);
    wav, lenth = ap.process_wav(wav, sr, sr)
    ap.outputAudio(out_test_o, wav, sr);
    #feat = fet.extract_features(wavfile, wav, sr)
    # Check feature size
    if len(feat) != EXPECTED_FEATURES:
        print(f"**Warning: {filepath} has {len(feat)} features instead of {EXPECTED_FEATURES}.")    
    #if len(feat)==0:
    #   print("May be empty frequency found")
    #print(feat)

if __name__ == "__main__":  # Ensure this runs only when executing main.py

    
   
    okay = process_extractFeat_ravdess_dataset()
    if okay == 0:
        print('Failed ravdess dataset')
        sys.exit(0)
    print('Completed ravdess dataset')

    okay = process_extractFeat_tess_dataset()
    if okay == 0:
        print('Failed tess dataset')
        sys.exit(0)
    print('Completed tess dataset')

    okay = process_extractFeat_savee_dataset()
    if okay == 0:
        print('Failed savee dataset')
        sys.exit(0)
    print('Completed savee dataset')

    okay = process_extractFeat_cremaD_dataset()
    if okay == 0:
        print('Failed cremaD dataset')
        sys.exit(0)
    print('Completed cremaD dataset')

    okay = process_extractFeat_emodb_dataset()
    if okay == 0:
        print('Failed Emodb dataset')
        sys.exit(0)
    print('Completed EmoDB dataset')

    sys.exit(0)
        
            
    

    
