import sys
sys.path.append('../../')
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
from timer import timer

from utils import DataTable
from utils import get_predictor

def get_voc_output(voc_predictor, input):
    voc_input_names = voc_predictor.get_input_names()
    mel_handle = voc_predictor.get_input_handle(voc_input_names[0])
    mel_handle.reshape(input.shape)
    mel_handle.copy_from_cpu(input)

    voc_predictor.run()
    voc_output_names = voc_predictor.get_output_names()
    voc_output_handle = voc_predictor.get_output_handle(voc_output_names[0])
    wav = voc_output_handle.copy_to_cpu()
    return wav

def main():
    test_metadata = './dump/test/raw/metadata.jsonl'

    # '''pwgan
    model_file = 'pwgan_csmsc.pdmodel'
    params_file = 'pwgan_csmsc.pdiparams'
    ## 非量化
    model_dir = './pwgan_csmsc_static_1.4.0'
    precision = 'fp32'
    ## 量化
    # model_dir = './pwgan_csmsc_quant'
    # precision = 'int8'
    # '''

    ''' mb_melgan
    model_file = 'mb_melgan_csmsc.pdmodel'
    params_file = 'mb_melgan_csmsc.pdiparams'
    ## 非量化
    model_dir = './mb_melgan_csmsc_static_1.4.0'
    precision = 'fp32'
    ## 量化
    # model_dir = './mb_melgan_csmsc_quant'
    # precision = 'int8'
    
    '''

    ''' hifigan
    model_file = 'hifigan_csmsc.pdmodel'
    params_file = 'hifigan_csmsc.pdiparams'
    ## 非量化
    model_dir = './hifigan_csmsc_static_1.4.0'
    precision = 'fp32'
    ## 量化
    # model_dir = './hifigan_csmsc_quant'
    # precision = 'int8'

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

    voc_predictor = get_predictor(
        model_dir=model_dir,
        model_file=model_file,
        params_file=params_file,
        device=device,
        use_mkldnn=True,
        precision=precision
    )
    
    # warm up
    for i, example in enumerate(test_dataset):
        if i < 3:
            utt_id = example['utt_id']
            mel = example['feats']
            with timer() as t:
                wav = get_voc_output(voc_predictor=voc_predictor, input=mel)

    for i, example in enumerate(test_dataset):
        # 太慢了，只跑 10 条, 对于 pwgan 可以改的更小，甚至只用 warmup
        if i < 10:
            utt_id = example['utt_id']
            mel = example['feats']
            with timer() as t:
                wav = get_voc_output(voc_predictor=voc_predictor, input=mel)
                N += wav.size
                T += t.elapse
                speed = wav.size / t.elapse
                rtf = fs / speed
            print(
                f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
            )
            sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=fs)
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")
    print(model_name, "inference with precision", precision, "success!" )
    

if __name__ == "__main__":
    main()


