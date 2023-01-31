量化模型导出失败
```bash
# 下载并解压缩输入数据
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/fastspeech2/dump.zip
unzip dump.zip
# fastspeech2 default
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/fastspeech2/fastspeech2_csmsc_static_1.4.0.zip
unzip fastspeech2_csmsc_static_1.4.0.zip
## 下载并解压缩量化模型
https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/fastspeech2/fastspeech2_csmsc_quant.zip
unizp fastspeech2_csmsc_quant.zip
# fastspeech2_cnndecoder
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/fastspeech2/fastspeech2_cnndecoder_csmsc_static_1.4.0.zip
unzip fastspeech2_cnndecoder_csmsc_static_1.4.0.zip
## !!量化模型导出报错暂不存在

```

执行如下命令进行静态图推理
```bash
python3 fastspeech2_unit_test.py
```
- 默认执行非量化模型的 mkldnn-fp32 推理, 执行成功
- **注释掉 34、35 行代码，并解开 37、38 行代码**后, 执行量化模型的 mkldnn-int8 推理, 推理报错