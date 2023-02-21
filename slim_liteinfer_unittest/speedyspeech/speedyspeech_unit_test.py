import sys
sys.path.append('../../')
from pathlib import Path

import jsonlines
import numpy as np
import paddle
from timer import timer

from utils import DataTable
from utils import get_lite_predictor


# speedyspeech
def get_lite_am_output(
        phone_ids,
        tone_ids,
        am_predictor: paddle.nn.Layer):
    phones = np.array(phone_ids)
    phones_handle = am_predictor.get_input(0)
    phones_handle.from_numpy(phones)
    tones = np.array(tone_ids)
    tones_handle = am_predictor.get_input(1)
    tones_handle.from_numpy(tones)
    am_predictor.run()
    am_output_handle = am_predictor.get_output(0)
    am_output_data = am_output_handle.numpy()
    return am_output_data

def main():
    test_metadata = './dump/test/norm/metadata.jsonl'
    model_file = 'speedyspeech_csmsc_x86.nb'
    # 非量化模型
    # model_dir = './speedyspeech_csmsc_pdlite_1.4.0'
    # precision = 'fp32'
    # 量化模型
    model_dir = './speedyspeech_csmsc_pdlite_quant'
    precision = 'int8'
    

    with jsonlines.open(test_metadata, 'r') as reader:
        metadata = list(reader)
    test_dataset = DataTable(
        metadata,
        fields=['utt_id', 'phones', 'tones'])

    N = 0
    T = 0
    fs = 24000
    device='cpu'

    model_name = model_file.split("/")[-1].split(".")[0]
    output_name = model_name + '_output'
    output_dir = Path(output_name) / precision
    output_dir.mkdir(parents=True, exist_ok=True)

    am_predictor = get_lite_predictor(
        model_dir=model_dir, model_file=model_file)

    # warm up
    for i, example in enumerate(test_dataset):
        if i <= 3:
            utt_id = example['utt_id']
            phone_ids = example['phones']
            tone_ids = example['tones']
            mel = get_lite_am_output(phone_ids=phone_ids, tone_ids=tone_ids, am_predictor=am_predictor)
    print("warm up done!")

    for i, example in enumerate(test_dataset):
        utt_id = example['utt_id']
        phone_ids = example['phones']
        tone_ids = example['tones']
        with timer() as t:
            mel = get_lite_am_output(phone_ids=phone_ids, tone_ids=tone_ids, am_predictor=am_predictor)
            print("mel.shape:", mel.shape)
            wav_size = mel.shape[0] * 300
            N += wav_size
            T += t.elapse
            speed = wav_size / t.elapse
            rtf = fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav_size}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        np.save(output_dir/(utt_id + ".npy"),mel)
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")
    print(model_name, "lite inference with precision", precision, "success!" )

if __name__ == "__main__":
    main()