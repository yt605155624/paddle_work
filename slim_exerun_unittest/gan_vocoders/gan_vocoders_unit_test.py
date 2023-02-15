import sys
sys.path.append('../../')
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
from timer import timer

from utils import DataTable


def main():
    test_metadata = './dump/test/raw/metadata.jsonl'

    # '''pwgan
    ## 非量化
    model_dir = './pwgan_csmsc_static_1.4.0/pwgan_csmsc'
    precision = 'fp32'
    ## 量化
    # model_dir = './pwgan_csmsc_quant/pwgan_csmsc'
    # precision = 'int8'
    # '''

    ''' mb_melgan
    ## 非量化
    # model_dir = './mb_melgan_csmsc_static_1.4.0/mb_melgan_csmsc'
    # precision = 'fp32'
    ## 量化
    # model_dir = './mb_melgan_csmsc_quant/mb_melgan_csmsc'
    # precision = 'int8'
    
    '''

    # ''' hifigan
    ## 非量化
    # model_dir = './hifigan_csmsc_static_1.4.0/hifigan_csmsc'
    # precision = 'fp32'
    ## 量化
    # model_dir = './hifigan_csmsc_quant/hifigan_csmsc'
    # precision = 'int8'
    # '''
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    [voc_inference, feed_target_names, fetch_targets] = paddle.static.load_inference_model(model_dir, exe)
   
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

    model_name = model_dir.split("/")[-1]
    output_name = model_name + '_output'
    output_dir = Path(output_name) / precision
    output_dir.mkdir(parents=True, exist_ok=True)

    # warm up
    for i, example in enumerate(test_dataset):
        if i < 3:
            utt_id = example['utt_id']
            mel = example['feats']
            wav = exe.run(voc_inference, feed={feed_target_names[0]:mel},fetch_list=fetch_targets)
    print("warm up done!")

    for i, example in enumerate(test_dataset):
        # 太慢了，只跑 10 条, 对于 pwgan 可以改的更小，甚至只用 warmup
        # if i < 10:
        utt_id = example['utt_id']
        mel = example['feats']
        with timer() as t:
            wav = exe.run(voc_inference, feed={feed_target_names[0]:mel},fetch_list=fetch_targets)
            wav = np.array(wav)[0]
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