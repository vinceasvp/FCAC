# FCAC

Few-shot class-incremental audio  classification, which can continually recognize novel audio classes without forgetting old ones

## Datasets

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, we constructed the LS-100 dataset, NSynth-100 dataset and FSC-89 dataset using partial samples from the Librispeech dataset, the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset as the source materials, respectively.

Wei Xie, one of our team members, constructed the NSynth-100 dataset and FSC-89 dataset. The detailed information of these two datasets is [here](https://github.com/chester-w-xie/FCAC_datasets).

The detailed information of the LS-100 dataset is given below.

### Statistics on the LS-100 dataset

|                                                                 | LS-100                                        |
|:---------------------------------------------------------------:|:---------------------------------------------:|
| Type of audio                                                   | Speech                                        |
| Num. of classes                                                 | 100 (60 of base classes, 40 of novel classes) |
| Num. of training / validation / testing samples per base class  | 500 / 150 / 100                               |
| Num. of training / validation / testing samples per novel class | 500 / 150 / 100                               |
| Duration of the sample                                          | All in 2 seconds                              |
| Sampling frequency                                              | All in 16K Hz                                 |

### Preparation of the LS-100 dataset

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We find that the subset ``train-clean-100`` of   Librispeech is enough for our study, so we constructed the LS-100 dataset using partial samples from the Librispeech as the source materials. To be specific, we first concatenate all the speakers' speech clips into a long speech, and then select the 100 speakers with the longest duration to cut their voices into two second speech. You can download the Librispeech from [here](https://www.openslr.org/12/).

1. Download [dataset](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract the files.
2. Transfer the format of audio files. Move the script ``normalize-resample.sh`` to the root dirctory of extracted folder, and run the command ``bash normalize-resample.sh``.
3. Construct LS-100 dataset.
   
   ```
   python data/LS100/construct_LS100.py --data_dir DATA_DIR --duration_json data/librispeech/spk_total_duration.json --single_spk_dir SINGLE_SPK_DIR --num_select_spk 100 --spk_segment_dir SPK_SEGMENT_DIR --csv_path CSV_PATH --spk_mapping_path SPK_MAPPING_PATH
   ```

## Code

To be sorted...
