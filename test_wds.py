from datasets import load_dataset

imagenet = load_dataset(
    "timm/imagenet-1k-wds",
    num_proc=8,
    data_files={
        "train": "imagenet1k-train-0000.tar",
    },
    split="train",
)
recap = load_dataset("Lucasdegeorge/ImageNet_TA_IA", split="train")
key_to_caption = {item["image_path"].split(".")[0]: item["caption"] for item in recap}

imagenet = imagenet.map(
    lambda x: {"caption": key_to_caption.get(x["__key__"], "")},
    num_proc=8,
)

for item in imagenet:
    break

print(item)
