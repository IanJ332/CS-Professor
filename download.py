from datasets import load_dataset

# 加载 Alpaca 数据集
print("Loading dataset...")
dataset = load_dataset('yahma/alpaca-cleaned')

print("Dataset loaded. Saving to disk...")
# 保存数据集
dataset.save_to_disk('./alpaca-dataset')

print("Dataset saved to ./alpaca-dataset.")
