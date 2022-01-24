from torch.utils.data import DataLoader
import torch

from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

from tqdm import tqdm

from custom_dataloader import MyCustomDataset2

def fine_tunned_model(path_to_folder, num_epochs = 5, bs=8):
    """ Function to fine-tunned the pretrained model on the dataset."""

    train_dataset = MyCustomDataset2(path_to_folder)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained("lincoln/flaubert-mlsum-topic-classification")
    model = AutoModelForSequenceClassification.from_pretrained("lincoln/flaubert-mlsum-topic-classification")

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            optim.zero_grad()
            inputs = tokenizer(batch["input_ids"], return_tensors="pt",device=device)
            print("inputs")
            print(inputs)

            #labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            labels = batch['labels'].to(device)
            print("labels")
            print(labels)
            break
            outputs = model(**inputs,labels = labels)

            loss = outputs[0]
            loss.backward()
            optim.step()
        break

    return model