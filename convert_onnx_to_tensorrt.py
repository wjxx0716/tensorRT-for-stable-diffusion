import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
from pathlib import Path
import argparse
import onnx
import ctypes

# 设置日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def get_current_gpu():
    """获取当前GPU信息，直接调用CUDA驱动，不依赖pycuda API"""
    try:
        # 加载CUDA驱动库
        if os.name == 'nt':  # Windows
            cuda_lib = ctypes.CDLL("nvcuda.dll")
        else:  # Linux/macOS
            cuda_lib = ctypes.CDLL("libcuda.so")
        
        # 初始化CUDA驱动
        result = cuda_lib.cuInit(0)
        if result != 0:
            return f"CUDA驱动初始化失败，错误码: {result}"
        
        # 获取GPU数量
        device_count = ctypes.c_int()
        cuda_lib.cuDeviceGetCount(ctypes.byref(device_count))
        
        # 获取当前设置的CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        visible_gpus = cuda_visible.split(',')
        
        # 验证每个GPU ID是否有效
        valid_gpus = []
        for gpu_id_str in visible_gpus:
            try:
                gpu_id = int(gpu_id_str)
                if 0 <= gpu_id < device_count.value:
                    valid_gpus.append(gpu_id)
            except ValueError:
                continue
        
        # 如果没有有效GPU ID，使用第一个GPU
        if not valid_gpus:
            selected_gpu_id = 0
        else:
            selected_gpu_id = valid_gpus[0]
        
        # 获取GPU名称
        name_buffer = ctypes.create_string_buffer(256)
        cuda_lib.cuDeviceGetName(
            ctypes.byref(name_buffer), 
            256, 
            selected_gpu_id
        )
        gpu_name = name_buffer.value.decode('utf-8')
        
        return f"GPU {selected_gpu_id}: {gpu_name} (系统共有 {device_count.value} 个GPU)"
        
    except Exception as e:
        return f"获取GPU信息失败: {str(e)}"

def build_engine(onnx_path, engine_path, fp16_mode=True, dynamic_shapes=None, workspace_size=4096):
    """构建 TensorRT 引擎"""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    
    # 创建构建器和网络
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # 设置工作空间大小（兼容新旧 API）
    if hasattr(config, 'max_workspace_size'):
        config.max_workspace_size = workspace_size * 1024 * 1024  # MB
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * 1024 * 1024)  # MB
    
    # 配置 FP16 模式
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 解析 ONNX 文件
    try:
        with open(onnx_path.absolute(), 'rb') as model:
            parser = trt.OnnxParser(network, TRT_LOGGER)
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # 自动标记输出层
        if network.num_outputs == 0:
            if network.num_layers > 0:
                last_layer = network.get_layer(network.num_layers - 1)
                for i in range(last_layer.num_outputs):
                    network.mark_output(last_layer.get_output(i))
                    print(f"Auto-marked output: {last_layer.get_output(i).name}")
            else:
                print("Error: Network has no layers or outputs")
                return None
    
    except FileNotFoundError:
        print(f"Error: Cannot find ONNX file at {onnx_path.absolute()}")
        return None
    
    # 设置动态形状
    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    
    # 构建并保存引擎
    print(f"Building TensorRT engine for {onnx_path}...")
    try:
        # 新版本 API (TensorRT 8.5+)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build serialized engine!")
            print(f"Network has {network.num_outputs} outputs")
            return None
        
        with open(engine_path.absolute(), 'wb') as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_path.absolute()}")
        return serialized_engine
    
    except AttributeError:
        # 旧版本 API
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build engine!")
            print(f"Network has {network.num_outputs} outputs")
            return None
        
        with open(engine_path.absolute(), 'wb') as f:
            f.write(engine.serialize())
        print(f"Engine saved to {engine_path.absolute()}")
        return engine

def convert_all_components(onnx_dir, trt_dir, fp16=True):
    """转换Stable Diffusion核心组件为TensorRT引擎（包含VAE Encoder）"""
    onnx_dir = Path(onnx_dir).absolute()
    trt_dir = Path(trt_dir).absolute()
    trt_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义全组件的动态形状配置
    dynamic_shapes_config = {
        "unet": {
            "sample": ((1, 4, 64, 64), (2, 4, 64, 64), (4, 4, 64, 64)),
            "timestep": ((1,), (2,), (4,)),
            "encoder_hidden_states": ((1, 77, 768), (2, 77, 768), (4, 77, 768))
        },
        "text_encoder": {
            "input_ids": ((1, 77), (2, 77), (4, 77))
        },
        "vae_decoder": {
            "latent_sample": ((1, 4, 64, 64), (2, 4, 64, 64), (4, 4, 64, 64))
        },
        "vae_encoder": {
            "sample": ((1, 3, 512, 512), (2, 3, 512, 512), (4, 3, 512, 512))
        }
    }
    
    # 转换UNet、Text Encoder、VAE Decoder和VAE Encoder
    components = ["unet", "text_encoder", "vae_decoder", "vae_encoder"]
    
    # 转换每个核心组件
    for component in components:
        onnx_path = onnx_dir / component / "model.onnx"
        trt_path = trt_dir / f"{component}.engine"
        
        if not onnx_path.exists():
            print(f"Skipping {component} as ONNX file not found at {onnx_path}")
            continue
        
        dynamic_shapes = dynamic_shapes_config.get(component, None)
        build_engine(
            onnx_path=str(onnx_path),
            engine_path=str(trt_path),
            fp16_mode=fp16,
            dynamic_shapes=dynamic_shapes
        )

def main():
    parser = argparse.ArgumentParser(description="Convert Stable Diffusion components to TensorRT engines")
    parser.add_argument("--onnx_dir", required=True, help="Directory containing ONNX models")
    parser.add_argument("--trt_dir", required=True, help="Output directory for TensorRT engines")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    args = parser.parse_args()
    
    # 打印环境信息
    print(f"pycuda版本: {get_pycuda_version()}")
    print(f"TensorRT版本: {trt.__version__}")
    
    # 打印当前使用的GPU信息
    gpu_info = get_current_gpu()
    print(f"当前环境使用的GPU: {gpu_info}")
    print(f"环境变量CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    
    convert_all_components(args.onnx_dir, args.trt_dir, args.fp16)
    
    try:
        unet_engine_path = os.path.join(args.trt_dir, "unet.engine")
        with open(unet_engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        # 检测引擎版本（兼容TensorRT 10.11）
        if hasattr(engine, 'serialization_version'):
            version = engine.serialization_version
        elif hasattr(engine, 'serialization_version_'):
            version = engine.serialization_version_
        else:
            # 使用tactic_sources或hardware_compatibility_level作为版本标识
            tactic_sources = engine.tactic_sources if hasattr(engine, 'tactic_sources') else "未知"
            hw_compat = engine.hardware_compatibility_level if hasattr(engine, 'hardware_compatibility_level') else "未知"
            version = f"TensorRT {trt.__version__} (策略源: {tactic_sources}, 硬件兼容性: {hw_compat})"
        
        print(f"引擎序列化版本: {version}")
        
    except Exception as e:
        print(f"读取引擎时出错: {e}")

if __name__ == "__main__":
    main()