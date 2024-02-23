"""
Split audio file based on VAD segments
"""

import os
import tqdm

import hydra
from omegaconf import DictConfig
from pydub import AudioSegment


@hydra.main(config_path="../conf", config_name="vad")
def split_audio(cfg: DictConfig):
    os.makedirs(cfg.split.out_dir, exist_ok=True)

    for txt_file in tqdm.tqdm(os.listdir(cfg.split.vad_segments)):
        audio_name = os.path.splitext(txt_file)[0]
        audio_filepath = os.path.join(cfg.vad.audio_dir, f"{audio_name}.wav")

        audio = AudioSegment.from_wav(audio_filepath)
        with open(os.path.join(cfg.split.vad_segments, txt_file), "r") as file:
            for idx, line in enumerate(file):
                split_line = line.split(",")
                t1 = float(split_line[0]) * 1000  # milliseconds
                t2 = float(split_line[1]) * 1000
                audio_segment = audio[t1:t2]
                new_filename = f"{audio_name}-{idx}.wav"

                # export audio file to output dir
                audio_segment.export(f"{cfg.split.out_dir}/{new_filename}")

            # ------- uncomment if want to remove original video -----------
            # try:
            #     os.remove(audio_filepath)
            #     print(f"Removed original video")
            # except Exception as e:
            #     print(f"Failed to remove original video, error = {e}")
            #     continue


if __name__ == "__main__":
    split_audio()
