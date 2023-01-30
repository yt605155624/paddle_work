量化模型导出失败
```bash
# 下载并解压缩输入数据
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/speedyspeech/dump.zip
unzip dump.zip

# 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/speedyspeech/speedyspeech_csmsc_static_1.4.0.zip
unzip speedyspeech_csmsc_static_1.4.0.zip
# !!量化模型导出报错暂不存在

```

执行如下命令进行静态图推理
```bash
python3 speedyspeech_unit_test.py
```
- 默认执行非量化模型的 mkldnn-fp32 推理, 执行成功
- **注释掉 41、42 行代码，并解开 45、46 行代码**后, 执行量化模型的 mkldnn-int8 推理, 推理报错