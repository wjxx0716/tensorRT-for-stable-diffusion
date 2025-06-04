import os
from pathlib import Path
import subprocess

def convert_sd_to_onnx(model_id="runwayml/stable-diffusion-v1-5", output_path="sd_onnx"):
    print("正在加载Stable Diffusion模型并转换为ONNX格式...")
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 下载ONNX转换脚本（如果不存在）
    script_path = "convert_stable_diffusion_checkpoint_to_onnx.py"
    if not os.path.exists(script_path):
        print("下载ONNX转换脚本...")
        subprocess.run([
            "wget", 
            "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py"
        ], check=True)
    
    # 执行转换命令
    print("开始转换模型组件...")
    subprocess.run([
        "python", script_path,
        "--model_path", model_id,
        "--output_path", str(output_path),
        "--opset", "17"
    ], check=True)
    
    print(f"转换完成！所有模型组件已保存到 {output_path} 目录")

if __name__ == "__main__":
    convert_sd_to_onnx()