# BASELINE TTS-ACOUSTIC MODEL FOR AISHELL-3 MULTI-SPEAKER MANDARIN CHINESE AUDIO CORPUS 
## Introduction
AISHELL-3 is a multi-speaker Mandarin Chinese audio corpus, this repository is the acoustic model for the multi-speaker TTS baseline system described in *AISHELL-3: A Multispeaker Mandarin Chinese TTS corpus* ([arXiv:2010.11567 [cs.SD]](https://arxiv.org/abs/2010.11567)).

Audio samples could be found [here](https://sos1sos2Sixteen.github.io/aishell3/index.html).
Dataset link on OpenSLR : [openslr/93](http://www.openslr.org/93/)

## Project Structure
### Model structure
`synthesizer`, `feedback_synthesizer` and `dca_synthesizer` defines the model architectures used in this project, all of which are extended tacotron-2 models and share the same file structure. 
* `synthesizer` is a plain multi-speaker tacotron-2 model, which uses 256-dimensional speaker embeddings as its speaker representation. 
* `dca_synthesizer` implements Dynamic Convolution Attention as a replacement to tacotron-2’s hybrid attetnion. 
* `feedback_synthesizer` implements speaker embedding feedback constraint on the acoustic model.
The speaker encoder network used in `feedback_synthesizer` is listed under `deep_speaker`.

### Scripts & Notebooks
* `process_audio.ipynb` is an off-line audio feature extraction script, which is used to build the `datasets` sub-directories.
* `synthesizer_train.py` & `fc_synthesizer_train.py`. We employs a two step strategy in training the baseline acoustic model: first we train a constraint-free model using `synthesizer_train.py`, then fine-tune the pre-trained model under feedback constraint using the same hyper-parameters with `fc_synthesizer_train.py`.
* `gvector_extraction.py` is used to batch inference speaker embeddings from Mel-spectrograms.
* `debug_syn.ipynb` shows the acoustic feature synthesis procedures using trained models.
* `vad.ipynb` & `longer_sentences.ipynb` are used to produce augmented training samples. `vad.ipynb` is used to trim initial silence segments from the mel-spectrograms using a naive energy based VAD approach. `longer_sentences.ipynb` produces longer training sentences by concatenating existing samples.

### Datasets
the `datasets` directory is intended to host training dataset data, one sub-directory for each separate dataset used in the experiment. But this intention was **not** hard-coded into the scripts, so feel free to do whatever you want, so long as the `dataset-directory` provided to the train scripts fullfills the requirements listed in the following usage notes.

A skeleton(incomplete) dataset directory is provided in the prject(`datasets/aishell3`). We provide in this directory the preprocessed train-set texts(with phoneme and prosodic labels) and averaged speaker embeddings as `metadata.csv` and `mean_embeddings` respectively.

## Usage
> replace `<name>` in the following code blocks with appropriate values.
> detailed usage of jupyter notebooks is described in the notebooks’ markdown blocks and comment sections.

### 0. Environment Setup
We use anaconda to manage our virtual environment. An exported conda env discription file is provided as `environment.yaml`. Use `conda` to create a new virtual environment in order to run the following scripts and notebooks: 

```
$ conda env create -f environment.yaml
```

This will create a new conda env named `aishell3`.

### 1. Synthesis with pre-trained model
1. Download the pre-trained checkpoints in this repository's [release page](https://github.com/sos1sos2Sixteen/aishell-3-baseline-fc/releases/tag/first_release); (checkpoints for a pretrained acoustic model and speaker encoder is provided here. For the pretrained WaveRNN model used in the synthesis demo(`debug_syn.ipynb`), please see [this repo](https://github.com/caizexin/tf_multispeakerTTS_fc) for information.)

2. use debug_syn.ipynb to load and inference the model

### 2. Train Speaker Encoder model
```
$cd deep_speaker
$CUDA_VISIBLE_DEVICES=<gpus> python train.py
```

### 3. Train Synthesizer (without feedback constraint)
1. Extract audio-features with `process_audio.ipynb`. An output directory named <dataset_name> should be specified within the notebook. (See the notebook’s content for more information).

2. (Optional) use `vad.ipynb` to trim initial silence segments in the extracted mel-spectrograms. We found this preprocess procedure helps speedup model convergence.

3. Extract speaker embeddings using `gvector_extraction.py`
```
$CUDA_VISIBLE_DEVICES=<gpu> python gvector_extraction.py <path-to-dataset-dir> --gvec_ckpt=<path-to-speaker-encoder-checkpoint>
```

4. Train base synthesizer, first set the proper batch-size and gpu-numbers in `synthesizer/hparams.py`:
```
# file: synthesizer/hprams.py
tacotron_num_gpus = <n_gpus>,
tacotron_batch_size = <bcsz>,
```

The training code supports data parallelism (samples within one logical batch are evenly spread among designated GPUs). We found that one 11G GTX1080Ti GPU could hold about 16\~24 samples per batch.

```
$CUDA_VISIBLE_DEVICES=<gpus> python synthesizer_train.py <run-name> <path-to-dataset>
```

**note**: the directory `<path-to-dataset>` should have the following sub-directories to correctly run the train script : 
```
<dataset>
    |- mels/            # generated by process_audio.ipynb or vad.ipynb
    |- embeds/      # generated by gvector_extraction.py
    |- train.txt        # generated by process_audio.ipynb
```

**note**: Modifications to `hparams.py` can also be passed to the train script using `--hparams` argument.

**note**: the optimization process could be monitored with tensorboard. the tensorboard events are being written to `synthesizer/saved_models/logs-<run_name>/tacotron_events` during the course of training.

5. Train feedback synthesizer using pre-trained base synthesizer parameters. First make sure `synthesizer/hparams.py` and `feedback_synthesizer/hparams.py` uses consistent model hyper-parameters(e.g. number of Pre-net layers etc.). Then set the pre-trained checkpoint path in hparams.py
```
# file: feedback_synthesizer/hparams.py
    
restore_tacotron_path = <path-to-pretrained-tacotron-checkpoint>
restore_spv_path = <path-to-pretrained-speaker-encoder-checkpoint>

```

```
$CUDA_VISIBLE_DEVICES=<gpus> python fc_synthesizer_train.py <run-name> <path-to_dataset>
```


