import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertSentimentClassifier:
    def __init__(self, config: dict):
        bert_cfg = config.get("models", {}).get("bert", {})
        self.model_name = bert_cfg.get("model_name", "bert-base-uncased")
        self.num_labels = bert_cfg.get("num_labels", 5)
        self.epochs = bert_cfg.get("epochs", 3)
        self.batch_size = bert_cfg.get("batch_size", 32)
        self.lr = bert_cfg.get("learning_rate", 2e-5)
        self.warmup_steps = bert_cfg.get("warmup_steps", 100)
        self.max_length = bert_cfg.get("max_length", 256)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model: Optional[BertForSequenceClassification] = None
        self._trained = False

    def _init_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.model.to(self.device)

    def train(self, texts: list[str], labels: list[int], val_texts=None, val_labels=None):
        self._init_model()

        train_dataset = ReviewDataset(texts, labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
        )

        self.model.train()
        history = []

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbl = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=lbl)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)

                if (batch_idx + 1) % 50 == 0:
                    logger.info(
                        f"  Epoch {epoch+1}/{self.epochs} batch {batch_idx+1}/{len(train_loader)} "
                        f"loss={loss.item():.4f}"
                    )

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            epoch_info = {"epoch": epoch + 1, "train_loss": epoch_loss, "train_acc": epoch_acc}

            if val_texts and val_labels:
                val_acc = self._evaluate_split(val_texts, val_labels)
                epoch_info["val_acc"] = val_acc

            history.append(epoch_info)
            logger.info(f"  Epoch {epoch+1}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

        self._trained = True
        return history

    def evaluate(self, texts: list[str], labels: list[int]) -> dict:
        if not self._trained:
            raise RuntimeError("Model not trained yet")

        acc = self._evaluate_split(texts, labels)
        preds = self.predict(texts)

        return {"accuracy": acc, "predictions": np.array(preds)}

    def predict(self, texts: list[str]) -> list[int]:
        self.model.eval()
        dataset = ReviewDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=mask)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)

        return all_preds

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        dataset = ReviewDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def _evaluate_split(self, texts: list[str], labels: list[int]) -> float:
        preds = self.predict(texts)
        correct = sum(p == l for p, l in zip(preds, labels))
        return correct / len(labels)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, config: dict) -> "BertSentimentClassifier":
        instance = cls(config)
        instance.model = BertForSequenceClassification.from_pretrained(path)
        instance.tokenizer = BertTokenizer.from_pretrained(path)
        instance.model.to(instance.device)
        instance._trained = True
        logger.info(f"Model loaded from {path}")
        return instance
