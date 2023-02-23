数据:
在 `slim_infer_unittest/speedyspeech` 目录中下载好输入数据并**软链接**到本目录下，**或者** 按照如下步骤下准备输入数据。
模型:
按照如下步骤下载

```bash
# 下载并解压缩输入数据
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/speedyspeech/dump.zip
unzip dump.zip
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/speedyspeech/speedyspeech_csmsc_pdlite_1.4.0.zip
unzip speedyspeech_csmsc_pdlite_1.4.0.zip
## 下载并解压缩量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_liteinfer_unittest/speedyspeech/speedyspeech_csmsc_pdlite_quant.zip
unzip speedyspeech_csmsc_pdlite_quant.zip
```

执行如下命令进行 Paddle-Lite 推理
```bash
python3 speedyspeech_unit_test.py
```
默认执行**非量化**模型的 **x86** 推理，请调整 32 ~ 38 行的代码更换 量化 / 非量化 模型或 x86 /arm 架构