from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="./data", split="train")
print(dataset[0]["prompt"])
dataset.push_to_hub("omkar1799/annalaura-diffusion-dataset")
print("Dataset pushed to the hub!")