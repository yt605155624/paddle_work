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

执行如下命令进行静态图推理
```bash
python3 gan_vocoders_unit_test.py
```
- 默认执行 pwgan 非量化模型的 mkldnn-fp32 推理, 执行成功

## PWGAN
!! 速度有点慢
29 行 ~ 37 行对应的代码段
- 默认执行非量化模型的 mkldnn-fp32 推理, 执行成功
- **注释掉 33、34 行代码，并解开 36、37 行代码**后, 执行量化模型的 mkldnn-int8 推理, 推理**卡死（其实是巨慢无比）**

另有报错如下, 不知是否有影响
```bash
E0130 14:46:18.929539  3179 analysis_config.cc:678] There are unsupported operators in the configured quantization operator list. The unsupported operator is: conv2d_transpose
```
## MB_MelGAN
40 行 ~ 50 行对应的代码段
量化模型导出失败暂不处理

## HiFiGAN
52 行 ~ 62 行代码段

- 默认执行非量化模型的 mkldnn-fp32 推理, 执行成功
- **注释掉 56、57 行代码，并解开 59、60 行代码**后, 执行量化模型的 mkldnn-int8 推理, 推理报错
