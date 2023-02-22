在 `slim_infer_unittest/speedyspeech` 目录中下载好输入数据和模型并**软链接**到本目录下，**或者** 按照如下步骤下准备输入数据和模型。

```bash
# 下载并解压缩输入数据
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/speedyspeech/dump.zip
unzip dump.zip

# 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/speedyspeech/speedyspeech_csmsc_static_1.4.0.zip
unzip speedyspeech_csmsc_static_1.4.0.zip
# 下载并解压缩量化模型（合成结果有问题）
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/speedyspeech/speedyspeech_csmsc_quant.zip
unzip speedyspeech_csmsc_quant.zip
```

执行如下命令进行静态图的 `exe.run()` 推理
```bash
python3 speedyspeech_unit_test.py
```
- 默认执行非量化模型的 `exe.run()` 推理, 执行成功
- **注释掉 16、17 行代码，并解开 20、21 行代码**后, 执行量化模型的 `exe.run()` 推理, 可以跑通，**但是压缩地太狠了**。