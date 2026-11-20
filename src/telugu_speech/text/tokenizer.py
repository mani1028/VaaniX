"""Character tokenizer for Telugu plus Latin code-mix."""

from __future__ import annotations

from telugu_speech.text.unicode import to_nfc

PAD = "<pad>"
UNK = "<unk>"
BLANK = "<blank>"


class CharTokenizer:
    def __init__(self, alphabet: str) -> None:
        symbols = [PAD, UNK, BLANK]
        for char in to_nfc(alphabet):
            if char not in symbols and not char.isspace():
                symbols.append(char)
        self.symbols = symbols
        self._index = {symbol: index for index, symbol in enumerate(symbols)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        return cls("".join(texts))

    def encode(self, text: str) -> list[int]:
        return [self._index.get(char, self._index[UNK]) for char in to_nfc(text) if not char.isspace()]

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.symbols):
                pieces.append(UNK)
                continue
            symbol = self.symbols[token_id]
            if symbol in {PAD, BLANK}:
                continue
            pieces.append(symbol)
        return "".join(pieces)

    @property
    def blank_id(self) -> int:
        return self._index[BLANK]
