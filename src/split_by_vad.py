"""
Split audio file based on VAD segments
"""
import os
import tqdm
from dotenv import load_dotenv
from pydub import AudioSegment


def split_audio(vad_dir: str, audio_dir: str, outdir_dir: str):
    os.makedirs(outdir_dir, exist_ok=True)

    for txt_file in tqdm.tqdm(os.listdir(vad_dir)):
        audio_name = os.path.splitext(txt_file)[0]
        audio_filepath = os.path.join(audio_dir, f"{audio_name}.wav")

        audio = AudioSegment.from_wav(audio_filepath)
        with open(os.path.join(vad_dir, txt_file), "r") as file:
            for idx, line in enumerate(file):
                split_line = line.split(",")
                t1 = float(split_line[0]) * 1000  # milliseconds
                t2 = float(split_line[1]) * 1000
                audio_segment = audio[t1:t2]
                new_filename = f"{audio_name}-{idx}.wav"

                # export audio file to output dir
                audio_segment.export(f"{outdir_dir}/{new_filename}")

            try:
                os.remove(audio_filepath)
                print(f"Removed original video")
            except Exception as e:
                print(f"Failed to remove original video, error = {e}")
                continue


if __name__ == "__main__":
    load_dotenv()
    AUDIO_DIR = os.getenv("AUDIO_DIR")
    SPLIT_DIR = os.getenv("SPLIT_DIR")

    split_audio(
        vad_dir="./output",
        audio_dir=AUDIO_DIR,
        outdir_dir=SPLIT_DIR,
    )
