import sys
sys.path.append('../../')
from pathlib import Path

import jsonlines
import numpy as np
import paddle
from timer import timer

from utils import DataTable


def main():
    test_metadata = './dump/test/norm/metadata.jsonl'
    # fastspeech2_default
    # 非量化模型
    model_dir = './fastspeech2_csmsc_static_1.4.0/fastspeech2_csmsc'
    precision = 'fp32'

    # 量化模型
    # model_dir = './fastspeech2_csmsc_quant/fastspeech2_csmsc'
    # precision = 'int8'
    
    '''fastspeech2_cnndecoder
    # 非量化模型
    model_dir = './fastspeech2_cnndecoder_csmsc_static_1.4.0/fastspeech2_csmsc'
    precision = 'fp32'
    # 量化模型
    # model_dir = './fastspeech2_cnndecoder_csmsc_quant/fastspeech2_csmsc'
    precision = 'int8'
    '''

    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    [fastspeech2_inference, feed_target_names, fetch_targets] = paddle.static.load_inference_model(model_dir, exe)

    with jsonlines.open(test_metadata, 'r') as reader:
        metadata = list(reader)
    test_dataset = DataTable(
        metadata,
        fields=['utt_id', 'text'])

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
        if i <= 3:
            utt_id = example['utt_id']
            phone_ids = example['text']
            mel = exe.run(fastspeech2_inference, feed={feed_target_names[0]:phone_ids},fetch_list=fetch_targets)
    print("warm up done!")

    for i, example in enumerate(test_dataset):
        utt_id = example['utt_id']
        phone_ids = example['text']
        with timer() as t:
            mel = exe.run(fastspeech2_inference, feed={feed_target_names[0]:phone_ids},fetch_list=fetch_targets)
            mel = np.array(mel)[0]
            print("mel.shape:",mel.shape)
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
    print(model_name, "exe.run() with precision", precision, "success!" )

if __name__ == "__main__":
    main()
