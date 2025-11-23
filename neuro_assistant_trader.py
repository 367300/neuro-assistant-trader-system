# %%
# Путь к директории с документами
source_path = '/content/data/'
# %%
from huggingface_hub import login
import os
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it in your .env file or export it.")
login(HF_TOKEN, add_to_git_credential=True)
# %%
# Импорт класса для чтения файлов из директории
from llama_index.core import SimpleDirectoryReader

# Импорт класса для чтения DOCX файлов
from llama_index.readers.file import DocxReader

# Импорт класса для создания индекса знаний
from llama_index.core import KnowledgeGraphIndex

# Импорт класса для настроек
from llama_index.core import Settings

# Импорт класса для простого хранилища графов
from llama_index.core.graph_stores import SimpleGraphStore

# Импорт класса для контекста хранения
from llama_index.core import StorageContext

# Импорт класса для встраивания моделей Hugging Face
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Импорт класса для языковых моделей Hugging Face
from llama_index.llms.huggingface import HuggingFaceLLM

# Импорт классов для моделей и конфигураций PEFT (Parameter-Efficient Fine-Tuning)
from peft import PeftModel, PeftConfig

# Импорт классов для автоматической загрузки моделей и токенизаторов из Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Импорт библиотеки PyTorch для работы с тензорами и нейронными сетями
import torch

# Импорт класса для встраивания моделей Langchain
from llama_index.embeddings.langchain import LangchainEmbedding

# Импорт класса для конфигурации BitsAndBytes
from transformers import BitsAndBytesConfig

# Импорт класса для создания шаблонов промптов
from llama_index.core.prompts import PromptTemplate

# Импорт модуля для получения имени пользователя
import getpass

# Импорт модуля для работы с операционной системой
import os
# %%
def messages_to_prompt(messages):
    # Инициализация пустой строки для промпта
    prompt = ""

    # Перебор всех сообщений
    for message in messages:
        # Если роль сообщения 'system', добавляем его в промпт
        if message.role == 'system':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        # Если роль сообщения 'user', добавляем его в промпт
        elif message.role == 'user':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        # Если роль сообщения 'bot', добавляем его в промпт
        elif message.role == 'bot':
            prompt += f"<s>bot\n"

    # Если промпт не начинается с 'system', добавляем его в начало
    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt

    # Добавляем 'bot' в конец промпта
    prompt = prompt + "<s>bot\n"

    # Возвращаем сформированный промпт
    return prompt

def completion_to_prompt(completion):
    # Формируем промпт для завершения, добавляя 'system', 'user' и 'bot'
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"

# %%
# Задаем имя модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
# %%
# Определяем параметры квантования, иначе модель не выполнится в колабе
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float8_e5m2,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
# %%
# Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
config = PeftConfig.from_pretrained(MODEL_NAME)

# Загружаем базовую модель, ее имя берем из конфига для LoRA
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,          # идентификатор модели
    quantization_config=quantization_config, # параметры квантования
    torch_dtype=torch.float16,               # тип данных
    device_map="auto"                        # автоматический выбор типа устройства
)

# Загружаем LoRA модель
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Переводим модель в режим инференса
model.eval()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# %%
# Сохраним модель после квантования для последующего использования.

output_dir = "/home/sigma/projects/neuro-assistant-trader-system/models/saiga_mistral_7b_e5m2"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
# %%
# Загрузить квантованную модель
model = AutoModelForCausalLM.from_pretrained(output_dir)

# Загрузить токенизатор
tokenizer = AutoTokenizer.from_pretrained(output_dir)
# %%
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)
# %%
print(123)
# %%
