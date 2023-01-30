
import sys
sys.path.append('../')
from pathlib import Path

import jsonlines
import numpy as np
import paddle
from timer import timer

from utils import DataTable
from utils import get_predictor

def get_am_output(
        phone_ids,
        tone_ids,
        am_predictor: paddle.nn.Layer):
    am_input_names = am_predictor.get_input_names()
    phones = np.array(phone_ids)
    phones_handle = am_predictor.get_input_handle(am_input_names[0])
    phones_handle.reshape(phones.shape)
    phones_handle.copy_from_cpu(phones)

    tones = np.array(tone_ids)
    tones_handle = am_predictor.get_input_handle(am_input_names[1])
    tones_handle.reshape(tones.shape)
    tones_handle.copy_from_cpu(tones)

    am_predictor.run()
    am_output_names = am_predictor.get_output_names()
    am_output_handle = am_predictor.get_output_handle(am_output_names[0])
    am_output_data = am_output_handle.copy_to_cpu()
    return am_output_data

def main():
    test_metadata = './dump/test/norm/metadata.jsonl'
    model_file = 'speedyspeech_csmsc.pdmodel'
    params_file = 'speedyspeech_csmsc.pdiparams'

    # 非量化模型
    model_dir = './speedyspeech_csmsc_static_1.4.0'
    precision = 'fp32'
    
    # 量化模型
    # model_dir = './speedyspeech_csmsc_quant'
    # precision = 'int8'

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

    am_predictor = get_predictor(
        model_dir=model_dir,
        model_file=model_file,
        params_file=params_file,
        device=device,
        use_mkldnn=True,
        precision=precision
    )

     # warm up
    for i, example in enumerate(test_dataset):
        if i <= 3:
            utt_id = example['utt_id']
            phone_ids = example['phones']
            tone_ids = example['tones']
            with timer() as t:
                mel = get_am_output(phone_ids=phone_ids,  tone_ids=tone_ids, am_predictor=am_predictor)
    print("warm up done!")


    for example in test_dataset:
        utt_id = example['utt_id']
        phone_ids = example['phones']
        tone_ids = example['tones']
        with timer() as t:
            mel = get_am_output(phone_ids=phone_ids, tone_ids=tone_ids, am_predictor=am_predictor)
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
    print(model_name, "inference with precision", precision, "success!" )

if __name__ == "__main__":
    main()