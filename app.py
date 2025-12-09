import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import importlib

# 1. 手动导入 Qwen 配置
try:
    # 尝试直接导入
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig

    print("成功导入 Qwen2_5OmniConfig")
except ImportError:
    print("无法导入 Qwen2_5OmniConfig，尝试修复...")

    # 2. 检查文件是否存在
    import sys
    import os

    # transformers 的安装路径
    import transformers

    transformers_path = os.path.dirname(transformers.__file__)
    qwen_config_path = os.path.join(transformers_path, "models", "qwen2_5_omni", "configuration_qwen2_5_omni.py")

    print(f"检查文件: {qwen_config_path}")
    print(f"文件存在: {os.path.exists(qwen_config_path)}")

    if os.path.exists(qwen_config_path):
        # 3. 手动执行模块
        import importlib.util

        spec = importlib.util.spec_from_file_location("qwen2_5_omni_config", qwen_config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 注册到 AutoConfig
        AutoConfig.register("qwen2_5_omni", module.Qwen2_5OmniConfig)
        print("手动注册成功！")
    else:
        print("文件不存在，需要重新安装 transformers")

# 现在尝试加载
config = AutoConfig.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    trust_remote_code=True
)
print(f"配置类: {config.__class__}")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    trust_remote_code=True,
    device_map="auto"
)