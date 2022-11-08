import argparse
import os
import json
from glob import glob
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
import librosa


def cat_spk_audio(data_dir, duration_json, single_spk_dir, num_select_spk):
    wav_path_list = sorted(glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    if os.path.exists(duration_json):
        with open(duration_json, 'r') as json_file:
            spk_total_duration = json.load(json_file)
    selected_spk_id = list(spk_total_duration.keys())[:num_select_spk]
    selected_spk_td = list(spk_total_duration.values())[:num_select_spk]
    for spk in tqdm(selected_spk_id, total=len(selected_spk_id)):
        spk_audio = AudioSegment.empty()
        for wav_path in wav_path_list:
            if spk == wav_path.split('/')[-3]:
                spk_audio += AudioSegment.from_wav(wav_path)
        spk_audio.export(os.path.join(single_spk_dir, f"{spk}.wav"), format='wav')


    wav_path_list = sorted(glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    flac_path_list = sorted(glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
    for flac_path in flac_path_list:
        if os.path.exists(flac_path):
            os.remove(flac_path)
    if os.path.exists(duration_json):
        with open(duration_json, 'r') as json_file:
            spk_total_duration = json.load(json_file)
    else:
        spk_total_duration = {}
        spks = os.listdir(data_dir)
        for spk in spks:
            spk_total_duration[spk] = 0.0

        for wav_path in tqdm(wav_path_list):
            spk = wav_path.split('/')[-3]
            # if spk not in spk_total_duration.keys():
                # spk_total_duration[spk] = 0.0
            seg_audio, sr = librosa.load(wav_path, sr=None)
            spk_total_duration[spk] += len(seg_audio) / sr
        spk_total_duration = dict(sorted(spk_total_duration.items(), key = lambda x:x[1], reverse=True))
        print(spk_total_duration)
        with open(duration_json, 'w') as json_file:
            json.dump(spk_total_duration, json_file, indent=2)

    selected_spk_id = list(spk_total_duration.keys())[:num_select_spk]
    selected_spk_td = list(spk_total_duration.values())[:num_select_spk]
    print(selected_spk_id)


def cut_single_spk(single_spk_dir, spk_segment_dir, csv_path, 
                    spk_mapping_path, seg_duration=2000, num_train_seg=500, num_val_seg=150, num_test_seg=100):

    train_csv_path = os.path.join(csv_path, "librispeech_fscil_train.csv")
    val_csv_path = os.path.join(csv_path, "librispeech_fscil_val.csv")
    test_csv_path = os.path.join(csv_path, "librispeech_fscil_test.csv")
    single_spk_path_list = sorted(glob(os.path.join(single_spk_dir, "*.wav")))
    spk_seg_path_list = sorted(glob(os.path.join(spk_segment_dir, "*.wav")))

    # empty the seg dir
    if len(spk_seg_path_list) > 0:
        for spk_seg_path in spk_seg_path_list:
            os.remove(spk_seg_path)

    for sing_spk_path in tqdm(single_spk_path_list):
        spk = os.path.basename(sing_spk_path).split('.')[0]
        spk_audio = AudioSegment.from_wav(sing_spk_path)
        num_full_segs = len(spk_audio) // seg_duration
        for i in range(num_full_segs):
            tmp_audio = spk_audio[i * seg_duration: (i + 1) * seg_duration]
            tmp_audio.export(os.path.join(spk_segment_dir, f"{spk}_{i}.wav"), format='wav')


    spks_list = [os.path.basename(spk_path).split('.')[0] for spk_path in single_spk_path_list]
    spk_mapping = {}
    for i, spk in enumerate(spks_list):
        spk_mapping[spk] = i
    with open(spk_mapping_path, 'w') as json_flie:
        json.dump(spk_mapping, json_flie, indent=2)

    spk_seg_path_list = sorted(glob(os.path.join(spk_segment_dir, "*.wav")))
    train_fn_list = []
    train_spk_list = []
    train_label_list = []

    val_fn_list = []
    val_spk_list = []
    val_label_list = []

    test_fn_list = []
    test_spk_list = []
    test_label_list = []

    for spk in tqdm(spks_list):
        this_spk_seg_path_list = sorted(glob(os.path.join(spk_segment_dir, f"{spk}*.wav")))
        this_spk_seg_fn_list = [os.path.basename(seg_path) for seg_path in this_spk_seg_path_list]
        train_fn_list.extend(this_spk_seg_fn_list[:num_train_seg])
        val_fn_list.extend(this_spk_seg_fn_list[num_train_seg: num_train_seg + num_val_seg])
        test_fn_list.extend(this_spk_seg_fn_list[num_train_seg + num_val_seg: num_train_seg + num_val_seg + num_test_seg])

        train_spk_list.extend([spk] * num_train_seg)
        val_spk_list.extend([spk] * num_val_seg)
        test_spk_list.extend([spk] * num_test_seg)

        train_label_list.extend([spk_mapping[spk]] * num_train_seg)
        val_label_list.extend([spk_mapping[spk]] * num_val_seg)
        test_label_list.extend([spk_mapping[spk]] * num_test_seg)

    train_dict = {"filename": train_fn_list, "speaker_id": train_spk_list, "label": train_label_list}
    val_dict = {"filename": val_fn_list, "speaker_id": val_spk_list, "label": val_label_list}
    test_dict = {"filename": test_fn_list, "speaker_id": test_spk_list, "label": test_label_list}

    train_csv = pd.DataFrame(train_dict)
    val_csv = pd.DataFrame(val_dict)
    test_csv = pd.DataFrame(test_dict)

    train_csv.to_csv(train_csv_path, sep=',', index=False)
    val_csv.to_csv(val_csv_path, sep=',', index=False)
    test_csv.to_csv(test_csv_path, sep=',', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset")
    parser.add_argument('--duration_json', type=str, default="data/librispeech/spk_total_duration.json", 
                        help="Path to save json file of total duration of every speaker")
    parser.add_argument('--single_spk_dir', type=str, help="Path to save audio files of \
                            total duration of every speaker")
    parser.add_argument('--num_select_spk', type=str, help="Number of chosen speakers")
    parser.add_argument('--spk_segment_dir', type=str, help="Path to save audio segments of all speakers")
    parser.add_argument('--csv_path', type=str, help="Path to save train, val and test csv files")
    parser.add_argument('--spk_mapping_path', type=str, help="Path of json file which maps speaker ids to labels")
    args = parser.parse_args()

    cat_spk_audio(args.data_dir, args.duration_json, args.single_spk_dir, args.num_select_spk)

    cut_single_spk(args.single_spk_dir, args.spk_segment_dir, args.csv_path, args.spk_mapping_path)
