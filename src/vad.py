"""
VAD for audio clips
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

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


@hydra.main(config_path="../conf", config_name="vad")
def get_vad_segments(cfg: DictConfig):

    logging.info("loading model...")

    model = Model.from_pretrained(cfg.vad.model)
    model.to(torch.device("cuda"))

    pipeline = VoiceActivityDetection(segmentation=cfg.vad.model)
    pipeline.to(torch.device("cuda"))

    logging.info("loaded vad model!")
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    audio_files = os.listdir(cfg.vad.audio_dir)

    os.makedirs("/workspace/output", exist_ok=True)
    for filename in tqdm.tqdm(audio_files):
        curr_dir = os.path.join(cfg.vad.audio_dir, filename)
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


if __name__ == "__main__":
    get_vad_segments()
