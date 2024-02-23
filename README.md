# audio-vad-splitter
Split audio based on Pyannote's VAD

### Set up
Build docker image and run docker-compose.yaml

Create config file ./conf/vad.yaml
```
vad: # step 1: get vad segments
  model: /models/pyannote/segmentation-3.0/pytorch_model.bin
  audio_dir: /path/to/audio_dir

split: # step 2: split audio into vad segments
  vad_segments: /workspace/output
  out_dir: /path/to/out_dir
```