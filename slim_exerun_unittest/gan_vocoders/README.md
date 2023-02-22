
在 `slim_infer_unittest/gan_vocoders` 目录中下载好输入数据和模型并**软链接**到本目录下，**或者** 按照如下步骤下准备输入数据和模型。

```bash
# 下载并解压缩输入数据
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/dump.zip
unzip dump.zip
# pwgan
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/pwgan_csmsc_static_1.4.0.zip
unzip pwgan_csmsc_static_1.4.0.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/pwgan_csmsc_quant.zip
unzip pwgan_csmsc_quant.zip

# mb_melgan
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/mb_melgan_csmsc_static_1.4.0.zip
unzip mb_melgan_csmsc_static_1.4.0.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/mb_melgan_csmsc_quant.zip
unzip mb_melgan_csmsc_quant.zip

# hifigan
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/hifigan_csmsc_static_1.4.0.zip
unzip hifigan_csmsc_static_1.4.0.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/hifigan_csmsc_quant.zip
unzip hifigan_csmsc_quant.zip
```

执行如下命令进行静态图的 `exe.run()` 推理
```bash
python3 gan_vocoders_unit_test.py
```
- 默认执行 pwgan 非量化模型的 `exe.run()` 推理, 执行成功

17 行 ~ 24 行对应的代码段
- 默认执行非量化模型的 `exe.run()` 推理, 执行成功
- **注释掉 19、20 行代码，并解开 22、23 行代码**后, 执行量化模型的 `exe.run()` 推理, **推理成功**


## MB_MelGAN
26 行 ~ 34 行对应的代码段
- 默认执行非量化模型的 `exe.run()` 推理, 执行成功
- **注释掉 28、29 行代码，并解开 31、32 行代码**后, 执行量化模型的 `exe.run()` 推理, **推理失败, 证明 PaddleSlim 导出的模型有问题**

## HiFiGAN
36 行 ~ 43 行代码段
- 默认执行非量化模型的 `exe.run()` 推理, 执行成功
- **注释掉 38、39 行代码，并解开 41、42 行代码**后, 执行量化模型的 `exe.run()` 推理, **推理失败, 证明 PaddleSlim 导出的模型有问题**