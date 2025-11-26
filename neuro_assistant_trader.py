# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

# %%
# Проверка доступности GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    device = "cuda"
else:
    print("WARNING: CUDA not available, using CPU")
    device = "cpu"

# %%
# Используем версию модели в bfloat16
# RTX 3090 (compute capability 8.6) не поддерживает FP8, но поддерживает 4-bit/8-bit quantization
model_name = "ai-sage/GigaChat3-10B-A1.8B-bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# Настройка квантизации через BitsAndBytes (4-bit для максимальной экономии памяти)
# Это значительно уменьшит размер модели и ускорит работу на RTX 3090
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Используем 4-bit квантизацию для экономии памяти
    bnb_4bit_compute_dtype=torch.bfloat16,  # Тип данных для вычислений
    bnb_4bit_quant_type="nf4",  # Тип квантизации: nf4 (рекомендуется) или fp4
    bnb_4bit_use_double_quant=True,  # Двойная квантизация для лучшего качества
)

# Загрузка модели с квантизацией
# Это уменьшит размер модели примерно в 4 раза и ускорит инференс
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(model_name)

# %%
messages = [
    {"role": "user", "content": "Приветик, милый!"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

# %%
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=1000)

# %%
result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=False)
print(result)
# %%
