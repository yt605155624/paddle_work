from pathlib import Path

from typing import Optional
from typing import Optional
from multiprocessing import Manager
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from paddle.io import Dataset

import os
import paddle
from paddle import inference
from paddlelite.lite import create_paddle_predictor
from paddlelite.lite import MobileConfig


class DataTable(Dataset):
    """Dataset to load and convert data for general purpose.
    Args:
        data (List[Dict[str, Any]]): Metadata, a list of meta datum, each of which is composed of  several fields
        fields (List[str], optional): Fields to use, if not specified, all the fields in the data are used, by default None
        converters (Dict[str, Callable], optional): Converters used to process each field, by default None
        use_cache (bool, optional): Whether to use cache, by default False

    Raises:
        ValueError:
            If there is some field that does not exist in data. 
        ValueError:
            If there is some field in converters that does not exist in fields.
    """

    def __init__(self,
                 data: List[Dict[str, Any]],
                 fields: List[str]=None,
                 converters: Dict[str, Callable]=None,
                 use_cache: bool=False):
        # metadata
        self.data = data
        assert len(data) > 0, "This dataset has no examples"

        # peak an example to get existing fields.
        first_example = self.data[0]
        fields_in_data = first_example.keys()

        # check all the requested fields exist
        if fields is None:
            self.fields = fields_in_data
        else:
            for field in fields:
                if field not in fields_in_data:
                    raise ValueError(
                        f"The requested field ({field}) is not found"
                        f"in the data. Fields in the data is {fields_in_data}")
            self.fields = fields

        # check converters
        if converters is None:
            self.converters = {}
        else:
            for field in converters.keys():
                if field not in self.fields:
                    raise ValueError(
                        f"The converter has a non existing field ({field})")
            self.converters = converters

        self.use_cache = use_cache
        if use_cache:
            self._initialize_cache()

    def _initialize_cache(self):
        self.manager = Manager()
        self.caches = self.manager.list()
        self.caches += [None for _ in range(len(self))]

    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return a meta-datum given an index."""
        return self.data[idx]

    def _convert(self, meta_datum: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a meta datum to an example by applying the corresponding 
        converters to each fields requested.

        Args:
            meta_datum (Dict[str, Any]): Meta datum

        Returns:
            Dict[str, Any]: Converted example
        """
        example = {}
        for field in self.fields:
            converter = self.converters.get(field, None)
            meta_datum_field = meta_datum[field]
            if converter is not None:
                converted_field = converter(meta_datum_field)
            else:
                converted_field = meta_datum_field
            example[field] = converted_field
        return example

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an example given an index.
        Args:
            idx (int): Index of the example to get

        Returns:
            Dict[str, Any]: A converted example
        """
        if self.use_cache and self.caches[idx] is not None:
            return self.caches[idx]

        meta_datum = self._get_metadata(idx)
        example = self._convert(meta_datum)

        if self.use_cache:
            self.caches[idx] = example

        return example

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns
        -------
        int
            The length of the dataset
        """
        return len(self.data)

# inference
def get_predictor(
        model_dir: Optional[os.PathLike]=None,
        model_file: Optional[os.PathLike]=None,
        params_file: Optional[os.PathLike]=None,
        device: str='cpu',
        # for gpu
        use_trt: bool=False,
        # for trt
        use_dynamic_shape: bool=True,
        min_subgraph_size: int=5,
        # for cpu
        cpu_threads: int=1,
        use_mkldnn: bool=False,
        # for trt or mkldnn
        precision: int="fp32"):
    """
    Args:
        model_dir (os.PathLike): root path of model.pdmodel and model.pdiparams.
        model_file (os.PathLike): name of model_file.
        params_file (os.PathLike): name of params_file.
        device (str): Choose the device you want to run, it can be: cpu/gpu, default is cpu.
        use_trt (bool): whether to use TensorRT or not in GPU.
        use_dynamic_shape (bool): use dynamic shape or not in TensorRT.
        use_mkldnn (bool): whether to use MKLDNN or not in CPU.
        cpu_threads (int): num of thread when use CPU.
        precision (str): mode of running (fp32/fp16/bf16/int8).  
    """
    rerun_flag = False
    if device != "gpu" and use_trt:
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='gpu', but device == {}".
            format(precision, device))

    config = inference.Config(
        str(Path(model_dir) / model_file), str(Path(model_dir) / params_file))
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    if device == "gpu":
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if use_mkldnn:
            # fp32
            config.enable_mkldnn()
            if precision == "int8":
                config.enable_mkldnn_int8({
                    "conv2d_transpose", "conv2d", "depthwise_conv2d", "pool2d",
                    "transpose2", "elementwise_mul"
                })
                # config.enable_mkldnn_int8()
            elif precision in {"fp16", "bf16"}:
                config.enable_mkldnn_bfloat16()
            print("MKLDNN with {}".format(precision))
    if use_trt:
        if precision == "bf16":
            print("paddle trt does not support bf16, switching to fp16.")
            precision = "fp16"
        precision_map = {
            "int8": inference.Config.Precision.Int8,
            "fp32": inference.Config.Precision.Float32,
            "fp16": inference.Config.Precision.Half,
        }
        assert precision in precision_map.keys()
        pdtxt_name = model_file.split(".")[0] + "_" + precision + ".txt"
        if use_dynamic_shape:
            dynamic_shape_file = os.path.join(model_dir, pdtxt_name)
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                # for fastspeech2
                config.exp_disable_tensorrt_ops(["reshape2"])
                print("trt set dynamic shape done!")
            else:
                # In order to avoid memory overflow when collecting dynamic shapes, it is changed to use CPU.
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(10)
                config.collect_shape_range_info(dynamic_shape_file)
                print("Start collect dynamic shape...")
                rerun_flag = True

        if not rerun_flag:
            print("Tensor RT with {}".format(precision))
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[precision],
                use_static=True,
                use_calib_mode=False, )

    predictor = inference.create_predictor(config)
    return predictor

def get_lite_predictor(model_dir: Optional[os.PathLike]=None,
                       model_file: Optional[os.PathLike]=None,
                       cpu_threads: int=1):
    config = MobileConfig()
    config.set_model_from_file(str(Path(model_dir) / model_file))
    predictor = create_paddle_predictor(config)
    return predictor


def get_lite_voc_output(voc_predictor, input):
    mel_handle = voc_predictor.get_input(0)
    mel_handle.from_numpy(input)
    voc_predictor.run()
    voc_output_handle = voc_predictor.get_output(0)
    wav = voc_output_handle.numpy()
    return wav

