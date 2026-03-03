from pathlib import Path

from datasets import load_from_disk

from data_processing.format import SlothDatasetBuilder
from model_processing.load import LoadModel
from train_processing.train import UnslothTrainer

from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns

HomePath = Path(__file__).parent.absolute()
dataset_path = HomePath / 'dataset' / 'raw_cleaned_dataset'
save_form_path = HomePath / 'dataset' / 'formatted_cleaned_dataset'
save_build_path = HomePath / 'dataset' / 'build_dataset'

prompt_template = """ข้อความข่าวมีดังนี้:
{}

จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
ประเภท 1: ข่าวจริง
ประเภท 2: ข่าวปลอม

คำตอบ
คำตอบที่ถูกต้องคือ: ประเภท {}"""

dataset = load_from_disk(dataset_path)
model_loader = LoadModel("jojo-ai-mst/thai-opt350m-instruct")
model,tokenizer = model_loader.get_model()

builder = SlothDatasetBuilder(dataset=dataset,
                              selected_columns=["Title","Verification_Status"],
                              prompt=prompt_template,
                              selected_tokenizer=tokenizer)
formatted = builder.format()
# print(formatted[0])
max_token = builder.check_max_tokens()
max_token = 256

print(f"max token: {max_token}")

model_loader.max_length = max_token

builder.save_format(save_form_path.as_posix())

build = builder.build()
# print(build[0])
builder.save_build(save_build_path.as_posix())

# load full lora or Qlora if not Peft then create One else use one
model,tokenizer = model_loader.get_model_lora() if model_loader.is_peft(model) else model_loader.get_model_Qlora()

formatted_dataset = load_from_disk(save_form_path)
formatted_split = formatted_dataset.train_test_split(test_size=0.2)

train_dataset = formatted_split['train']
eval_dataset = formatted_split['test']

raw_dataset = load_from_disk(dataset_path)
raw_split = raw_dataset.train_test_split(test_size=0.2)

eval_raw = raw_split['test']


trainer_runner = UnslothTrainer(model=model,
               tokenizer=tokenizer,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               )
trainer_runner.train()

def evaluate_model(model, tokenizer, eval_raw):
    model.eval()
    preds = []
    labels = []

    for example in eval_raw:
        text = f"""ข้อความข่าวมีดังนี้:
{example['Title']}

จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
ประเภท 1: ข่าวจริง
ประเภท 2: ข่าวปลอม

คำตอบ
คำตอบที่ถูกต้องคือ:"""

        label = example["Verification_Status"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "ประเภท 1" in decoded:
            preds.append(1)
        elif "ประเภท 2" in decoded:
            preds.append(2)
        else:
            preds.append(-1)

        labels.append(label)

    acc = accuracy_score(labels, preds)
    print("Accuracy:", acc)

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

evaluate_model(model, tokenizer, eval_raw)

#TODO
# might need datacollation for train classication
# might use an unsloth's dataset gpt type conversation later.
# find best parameter with
# new feature
