"""Предобработка текста - та же, что в ноутбуке 2Gis_V2.ipynb.

Если поменять здесь хоть что-то, модель TF-IDF надо переобучить:
слова при обучении и при предсказании должны приводиться к одному виду.
"""
import re
from functools import lru_cache

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()

# предлоги, частицы, союзы, местоимения, междометия
STOP_POS = {"PREP", "PRCL", "CONJ", "NPRO", "INTJ"}
# отрицания оставляем, они меняют смысл ("не уютно")
NEGATIONS = {"не", "нет"}

_NON_LETTERS = re.compile(r"[^а-яёa-z\s]")


@lru_cache(maxsize=300_000)
def _lemma(word: str) -> str | None:
    if word in NEGATIONS:
        return word
    parsed = _morph.parse(word)[0]
    if parsed.tag.POS in STOP_POS:
        return None
    return parsed.normal_form


def preprocess_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = _NON_LETTERS.sub(" ", text.lower())
    lemmas = (_lemma(word) for word in text.split())
    return " ".join(lemma for lemma in lemmas if lemma)
