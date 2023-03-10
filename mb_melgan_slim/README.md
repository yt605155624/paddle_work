## MB_MelGAN 静态离线量化
确保已经按照 ../fs2_slim 的 readme 所示下载数据集并解压缩到 ~/datasets/BZNSYP

```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech/examples/csmsc/voc3
wget https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz
tar zxvf baker_alignment_tone.tar.gz
# 数据预处理，以下步骤速度较慢, 10 min 左右，可以在 tmux 中执行, 执行成功后会生成 dump 文件
./run.sh --stage 0 --stop-stage 0
## 下载并解压缩未量化模型
wget https://paddlespeech.bj.bcebos.com/demos/paddle_work/slim_infer_unittest/gan_vocoders/mb_melgan_csmsc_static_1.4.0.zip
unzip mb_melgan_csmsc_static_1.4.0.zip
mkdir -p exp/default/inference
# 未量化模型拷贝到指定位置
cp mb_melgan_csmsc_static_1.4.0/* exp/default/inference
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
exp/default/inference/mb_melgan_csmsc_quant
```

## 量化后的模型推理

拷贝**本 README 目录**下的 `mb_melgan_int8_infer.ipynb` 到 `PaddleSpeech/examples/csmsc/voc3`, 在 `PaddleSpeech/examples/csmsc/voc3` 中执行 ipynb 文件
