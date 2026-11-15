"""Compact BM25 candidate retriever over the index's 70k product metadata.

This is an integration-side metadata retriever, NOT a reconstruction of the
teammate's previous BM25 experiment implementation or its published scores.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import Any, Sequence

_TOKEN = re.compile(r'[^\W_]+', re.UNICODE)

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).casefold())

class BM25Index:
    def __init__(self, products: Sequence[Any]):
        self.ids = []
        self.docs: dict[str, list[tuple[int,int]]] = defaultdict(list)
        self.doc_lens = []
        for index, p in enumerate(products):
            get = p.get if isinstance(p, dict) else lambda k, d='': getattr(p, k, d)
            pid = str(get('product_id'))
            self.ids.append(pid)
            text = ' '.join(str(get(k, '')) for k in ('name', 'category_small','color','pattern'))
            bag = Counter(tokenize(text))
            self.doc_lens.append(sum(bag.values()))
            for term, tf in bag.items():self.docs[term].append((index, tf))
        self.avglen = sum(self.doc_lens) / len(self.ids) if self.ids else 1

    def search(self, query: str, top_k: int = 20) -> list[str]:
        if top_k < 1: return []
        n = len(self.ids)
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self.docs.get(term)
            if not posting: continue
            df = len(posting)
            idf = math.log(1+(n-df+0.5)/(df+0.5))
            for idx, tf in posting:
                norm = tf+1.5*(1-0.75+0.75*self.doc_lens[idx]/self.avglen)
                scores[idx] += idf*tf*2.5/norm
        return [self.ids[i] for i, _ in sorted(scores.items(), key=lambda x:(-x[1], x[0]))[:top_k]]
