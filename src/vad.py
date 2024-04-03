"""
VAD with splitting of audio clip
"""

import os
import csv
import logging
import tqdm
import torch
import hydra
from omegaconf import DictConfig
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


def get_vad_segments(vad_model: str, audio_dir: str):

    logging.info("loading model...")

    pipeline = VoiceActivityDetection(segmentation=vad_model)
    pipeline.to(torch.device("cuda"))

    logging.info("loaded vad model!")
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    audio_files = os.listdir(audio_dir)

    os.makedirs("/workspace/output", exist_ok=True)
    for filename in tqdm.tqdm(audio_files):
        curr_dir = os.path.join(audio_dir, filename)
        vid_name = os.path.splitext(filename)[0]

        with open(f"/workspace/output/{vid_name}.txt", "w") as out_file:
            writer = csv.writer(
                out_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            vad = pipeline(
                curr_dir
            )  # returns pyannote Annotation object (https://github.com/pyannote/pyannote-core/blob/develop/pyannote/core/annotation.py)

            # v_tracks returns sorted dictionary
            # Dict[Segment, Dict[TrackName, Label]] = SortedDict()
            # keys: annotated segments
            # values: {track: label} dictionary
            for segment, v in vad._tracks.items():
                writer.writerow([segment.start, segment.end, v])


def split_audio(out_dir: str, vad_segments: str, audio_dir: str):
    logging.info("Splitting audio files by vad segments...")

    os.makedirs(out_dir, exist_ok=True)

    for txt_file in tqdm.tqdm(os.listdir(vad_segments)):
        audio_name = os.path.splitext(txt_file)[0]
        audio_filepath = os.path.join(audio_dir, f"{audio_name}.wav")

        audio = AudioSegment.from_wav(audio_filepath)
        with open(os.path.join(vad_segments, txt_file), "r") as file:
            for idx, line in enumerate(file):
                split_line = line.split(",")
                t1 = float(split_line[0]) * 1000  # milliseconds
                t2 = float(split_line[1]) * 1000
                audio_segment = audio[t1:t2]
                new_filename = f"{audio_name}-{idx}.wav"

                # export audio file to output dir
                audio_segment.export(f"{out_dir}/{new_filename}")

        # ------- uncomment if want to remove original video -----------
        # try:
        #     os.remove(audio_filepath)
        #     print(f"Removed original video")
        # except Exception as e:
        #     print(f"Failed to remove original video, error = {e}")
        #     continue


@hydra.main(config_path="../conf", config_name="vad")
def main(cfg: DictConfig):
    get_vad_segments(vad_model=cfg.vad.model, audio_dir=cfg.vad.audio_dir)
    split_audio(
        out_dir=cfg.split.out_dir,
        vad_segments=cfg.split.vad_segments,
        audio_dir=cfg.vad.audio_dir,
    )


if __name__ == "__main__":
    main()
