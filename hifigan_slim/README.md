## MB_MelGAN 静态离线量化
确保已经按照 ../fs2_slim 的 readme 所示下载数据集并解压缩到 `~/datasets/BZNSYP`

确保已经按照 ../mb_melgan_slim 的 readme 所示执行了数据预处理，成功生成 `dump` 文件

```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech/examples/csmsc/voc5
# hifigan 的 dump 和 mb_melgan 一样，直接软链接即可
ln -snf ../voc3/dump .
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/hifigan_csmsc_static_1.4.0.zip
unzip hifigan_csmsc_static_1.4.0.zip
mkdir -p exp/default/inference
# 未量化模型拷贝到指定位置
cp hifigan_csmsc_static_1.4.0/* exp/default/inference
# 执行静态离线量化
./run.sh --stage 3 --stop-stage 3 
```

静态离线量化调用的代码是
```text
./local/PTQ_static.sh
```

python 文件是 
```text
PaddleSpeech/paddlespeech/t2s/exps/PTQ_static.py
```
可以修改这个文件的参数来调整静态离线量化的参数，如新增 `--quantizable_op_type "matmul" "matmul_v2"` 控制 quantizable_op_type，默认是使用所有 op_type

生成的量化模型在 
```text
exp/default/inference/hifigan_csmsc_quant
```

## 量化后的模型推理

拷贝**本 README 目录**下的 `hifigan_int8_infer.ipynb` 到 `PaddleSpeech/examples/csmsc/voc5`, 在 `PaddleSpeech/examples/csmsc/voc5` 中执行 ipynb 文件
