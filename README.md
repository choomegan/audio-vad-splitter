# audio-vad-splitter
Split audio based on Pyannote's VAD

## Set up
### Download model file
Download pyannote's segmentation-3.0 from https://huggingface.co/pyannote/segmentation-3.0. Put model files under /models folder

### Docker
Build docker image and run docker-compose.yaml

Create config file ./conf/vad.yaml
```
vad: # step 1: get vad segments
  model: /models/pyannote/segmentation-3.0/pytorch_model.bin
  audio_dir: /path/to/audio_dir

split: # step 2: split audio into vad segments
  vad_segments: /workspace/output
  out_dir: /path/to/split_dir

resample:
  out_dir: /path/to/resampled_dir
```

## VAD

Inside the container, run [vad.py]('src/vad.py') script. The script does the following functions: 

1. Uses pyannote model to get VAD segments
2. Splits audio in audio_dir according to these segments
3. Resamples split audio to mono channel, 16kHz sampling rate, 256k bit rate WAV files

VAD segments from step (1) will be saved to ./output

```
python3 src/vad.py
```

Toggle min_duration_on and min_duration_off for the VAD model as required.