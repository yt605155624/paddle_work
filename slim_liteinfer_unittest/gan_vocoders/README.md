数据:
在 `slim_infer_unittest/gan_vocoders` 目录中下载好输入数据并**软链接**到本目录下，**或者** 按照如下步骤下准备输入数据。
模型:
按照如下步骤下载

```bash
# 下载并解压缩输入数据
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/dump.zip
unzip dump.zip
# pwgan
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/gan_vocoders/pwgan_csmsc_pdlite_quant.zip
unzip pwgan_csmsc_pdlite_quant.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/gan_vocoders/pwgan_csmsc_pdlite_quant.zip
unzip pwgan_csmsc_pdlite_quant.zip
# mb_melgan
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/gan_vocoders/mb_melgan_csmsc_pdlite_1.4.0.zip
unzip mb_melgan_csmsc_pdlite_1.4.0.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/gan_vocoders/mb_melgan_csmsc_pdlite_quant.zip
unzip mb_melgan_csmsc_pdlite_quant.zip
# hifigan
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/gan_vocoders/hifigan_csmsc_pdlite_1.4.0.zip
unzip hifigan_csmsc_pdlite_1.4.0.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/gan_vocoders/hifigan_csmsc_pdlite_quant.zip
unzip hifigan_csmsc_pdlite_quant.zip
```
执行如下命令进行 Paddle-Lite 推理
```bash
python3 gan_vocoders_unit_test.py
```
默认执行 pwgan **非量化**模型的 **x86** 推理，请调整 19 ~ 48 行的代码更换 vocoder、量化 / 非量化 模型或 x86 /arm 架构