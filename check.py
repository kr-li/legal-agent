# 运行此脚本检查安装
import transformers
import os

print(f"transformers 版本: {transformers.__version__}")
print(f"安装路径: {transformers.__file__}")

# 检查 qwen2_5_omni 是否存在
transformers_path = os.path.dirname(transformers.__file__)
qwen_dir = os.path.join(transformers_path, "models", "qwen2_5_omni")

print(f"Qwen2.5-Omni 目录: {qwen_dir}")
print(f"目录是否存在: {os.path.exists(qwen_dir)}")

if os.path.exists(qwen_dir):
    files = os.listdir(qwen_dir)
    print(f"目录内容: {files}")

    # 检查配置文件
    config_file = os.path.join(qwen_dir, "configuration_qwen2_5_omni.py")
    print(f"配置文件: {config_file}")
    print(f"配置文件存在: {os.path.exists(config_file)}")