from datasets import load_dataset
from transformers import AutoTokenizer

# 选择你的模型名称
model_name = "meta-llama/Llama-3.5-405B"  # 替换为实际使用的模型名称

# 加载 Alpaca 数据集
dataset = load_dataset('yahma/alpaca-cleaned')

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# 处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 保存处理后的数据集（如果需要）
tokenized_datasets.save_to_disk('./tokenized_datasets')

print("Data preprocessing completed and saved to './tokenized_datasets'")
