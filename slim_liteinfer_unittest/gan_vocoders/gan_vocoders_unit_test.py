import sys
sys.path.append('../../')
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
from timer import timer

from utils import DataTable
from utils import get_lite_predictor
from utils import get_lite_voc_output



def main():
    test_metadata = './dump/test/raw/metadata.jsonl'
    
    # '''pwgan
    # 修改 x86 为 arm 后在 arm 环境下推理
    model_file = 'pwgan_csmsc_arm.nb'
    ## 非量化
    model_dir = './pwgan_csmsc_pdlite_1.4.0'
    precision = 'fp32'
    ## 量化
    # model_dir = './pwgan_csmsc_pdlite_quant'
    # precision = 'int8'
    # '''

    ''' mb_melgan
    model_file = 'mb_melgan_csmsc_arm.nb'
    ## 非量化
    model_dir = './mb_melgan_csmsc_pdlite_1.4.0'
    precision = 'fp32'
    ## 量化
    # model_dir = './mb_melgan_csmsc_pdlite_quant'
    # precision = 'int8'
    
    '''

    ''' hifigan
    model_file = 'hifigan_csmsc_arm.nb'
    ## 非量化
    # model_dir = './hifigan_csmsc_pdlite_1.4.0'
    # precision = 'fp32'
    ## 量化
    model_dir = './hifigan_csmsc_pdlite_quant'
    precision = 'int8'
    '''
    
    with jsonlines.open(test_metadata, 'r') as reader:
        metadata = list(reader)
    test_dataset = DataTable(
        metadata,
        fields=['utt_id', 'feats'],
        converters={
            'utt_id': None,
            'feats': np.load,
        })

    N = 0
    T = 0
    fs = 24000
    device='cpu'

    model_name = model_file.split("/")[-1].split(".")[0]
    output_name = model_name + '_output'
    output_dir = Path(output_name) / precision
    output_dir.mkdir(parents=True, exist_ok=True)

    voc_predictor = get_lite_predictor(
        model_dir=model_dir, model_file=model_file)

    # warm up
    print("warm up ...")
    for i, example in enumerate(test_dataset):
        if i <= 3:
            utt_id = example['utt_id']
            mel = example['feats']
            # to avoid segmentation fault for arm pwgan
            # 1/4 ok for fp32, 1/7 ok for int8
            if 'pwgan' in model_file:
                mel = mel[:mel.shape[0]//7]
            wav = get_lite_voc_output(voc_predictor=voc_predictor, input=mel)
    print("warm up done!")

    for i, example in enumerate(test_dataset):
        utt_id = example['utt_id']
        mel = example['feats']
        with timer() as t:
            # to avoid segmentation fault for arm pwgan
            # 1/4 ok for fp32, 1/7 ok for int8
            if 'pwgan' in model_file:
                mel = mel[:mel.shape[0]//7]
            wav = get_lite_voc_output(voc_predictor=voc_predictor, input=mel)
            wav_size = wav.size
            N += wav_size
            T += t.elapse
            speed = wav_size / t.elapse
            rtf = fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav_size}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=fs)
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")
    print(model_name, "lite inference with precision", precision, "success!" )

if __name__ == "__main__":
    main()