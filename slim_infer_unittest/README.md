fastspeech2 量化报错，暂时不用管

speedyspeech 量化报错，暂时不用管

gan_vocoders 中包含 3 个模型
- pwgan
- mb_melgan 量化报错，暂时不用管
- hifigan

cd 到对应目录按照 README 执行即可复现

模型输出均在对应文件夹下 `${model_name}`_output / `${precision}` 文件夹中, 其中 `${precision}` 可选 `fp32` 和 `int8`