import re
import logging
import math
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class ExtractiveSummarizer:
    """TF-IDF-weighted sentence scoring for extractive summarization."""

    def __init__(self, config: dict):
        summ_cfg = config.get("summarization", {})
        self.num_sentences = summ_cfg.get("num_sentences", 3)
        self.min_sentence_length = summ_cfg.get("min_sentence_length", 5)

    def summarize(self, text: str, num_sentences: int = None) -> str:
        n = num_sentences or self.num_sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= n:
            return text

        scores = self._score_sentences(sentences)

        # pick top-n by score but keep original order
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked_indices[:n])

        summary = " ".join(sentences[i] for i in selected)
        return summary

    def summarize_batch(self, texts: list[str], num_sentences: int = None) -> list[dict]:
        results = []
        for text in texts:
            summary = self.summarize(text, num_sentences)
            orig_len = len(text.split())
            summ_len = len(summary.split())
            ratio = summ_len / orig_len if orig_len > 0 else 1.0

            results.append({
                "original": text,
                "summary": summary,
                "compression_ratio": round(ratio, 3),
                "num_sentences": len(self._split_sentences(summary)),
            })
        return results

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = []
        for s in raw:
            s = s.strip()
            if len(s.split()) >= self.min_sentence_length:
                sentences.append(s)
        return sentences

    def _score_sentences(self, sentences: list[str]) -> list[float]:
        # build document frequency across sentences
        word_lists = [self._tokenize(s) for s in sentences]
        doc_freq: Counter = Counter()
        for words in word_lists:
            doc_freq.update(set(words))

        n_docs = len(sentences)
        scores = []

        for words in word_lists:
            if not words:
                scores.append(0.0)
                continue

            tf = Counter(words)
            score = 0.0
            for word, count in tf.items():
                tf_val = count / len(words)
                idf_val = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1
                score += tf_val * idf_val

            # normalize by sentence length to avoid favoring long sentences
            score /= math.sqrt(len(words))

            # small boost for position (first/last sentences tend to be important)
            idx = word_lists.index(words)
            if idx == 0:
                score *= 1.2
            elif idx == len(sentences) - 1:
                score *= 1.05

            scores.append(score)

        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # basic stop word removal
        stops = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "and", "but", "or", "not", "no", "it", "its",
            "this", "that", "these", "those", "i", "me", "my", "we", "our",
            "you", "your", "he", "him", "she", "her", "they", "them", "their",
        }
        return [w for w in words if w not in stops and len(w) > 2]
