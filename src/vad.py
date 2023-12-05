"""
VAD for audio clips
"""
import os
import csv
import tqdm
import torch
from dotenv import load_dotenv
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


def get_vad_segments(
    model: str,
    root_dir: str,
):
    """
    Expected folder structure of root_dir:
    root_dir / subdir  / audio file 1
             / subdir2 / audio file 2
    """

    model = Model.from_pretrained(model, use_auth_token=HF_AUTH_TOKEN)
    model.to(torch.device("cuda"))

    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.to(torch.device("cuda"))

    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    subdirs = os.listdir(root_dir)

    os.makedirs("./output", exist_ok=True)
    for folder_name in tqdm.tqdm(subdirs):
        curr_dir = os.path.join(root_dir, folder_name)
        vid_name = os.listdir(curr_dir)[0]
        audio_dir = os.path.join(curr_dir, vid_name)

        with open(f"./output/{os.path.splitext(vid_name)[0]}.txt", "w") as out_file:
            writer = csv.writer(
                out_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            vad = pipeline(
                audio_dir
            )  # returns pyannote Annotation object (https://github.com/pyannote/pyannote-core/blob/develop/pyannote/core/annotation.py)

            # v_tracks returns sorted dictionary
            # Dict[Segment, Dict[TrackName, Label]] = SortedDict()
            # keys: annotated segments
            # values: {track: label} dictionary
            for segment, v in vad._tracks.items():
                writer.writerow([segment.start, segment.end, v])


if __name__ == "__main__":
    load_dotenv()
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    AUDIO_DIR = os.getenv("AUDIO_DIR")

    get_vad_segments(
        model="pyannote/segmentation-3.0",
        root_dir=AUDIO_DIR,
    )
