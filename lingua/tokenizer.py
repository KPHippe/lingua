# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import os

try:
    from sentencepiece import SentencePieceProcessor

    has_sp = True
except ImportError:
    has_sp = False

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    has_tiktoken = True
except ImportError:
    has_tiktoken = False

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens, add_bos, add_eos):
        return tokens


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
        for id in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int]):
        return self.tkt_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(
                self.tkt_model.encode(text, allowed_special="all")
            )

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets


class AminoAcidTokenizer(Tokenizer):

    def __init__(self) -> None:
        # Define standard amino acids and their single-letter codes
        AMINO_ACIDS = [
            'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
        ]

        # Special tokens dictionary
        SPECIAL_TOKENS = {
            '<BOS>': 0,  # Beginning of sequence
            '<EOS>': 1,  # End of sequence
            '<UNK>': 2,  # Unknown token
            '<PAD>': 3,  # Padding
        }

        # Create token mappings, starting IDs after special tokens
        TOKEN_TO_ID = {aa: idx + len(SPECIAL_TOKENS) for idx, aa in enumerate(AMINO_ACIDS)}
        ID_TO_TOKEN = {idx + len(SPECIAL_TOKENS): aa for idx, aa in enumerate(AMINO_ACIDS)}

        # Include special tokens in the mappings
        TOKEN_TO_ID.update(SPECIAL_TOKENS)
        ID_TO_TOKEN.update({id_: token for token, id_ in SPECIAL_TOKENS.items()})

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.TOKEN_TO_ID = TOKEN_TO_ID
        self.ID_TO_TOKEN = ID_TO_TOKEN

        # Required to satisfy API
        self.bos_id = self.SPECIAL_TOKENS['<BOS>']
        self.eos_id = self.SPECIAL_TOKENS['<EOS>']
        self.n_words = len(TOKEN_TO_ID)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = []
        for char in text:
            token_id = self.TOKEN_TO_ID.get(char, self.SPECIAL_TOKENS['<UNK>'])
            tokens.append(token_id)
        if add_bos:
            tokens = [self.SPECIAL_TOKENS['<BOS>']] + tokens
        if add_eos:
            tokens = tokens + [self.SPECIAL_TOKENS['<EOS>']]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for token in tokens:
            token_str = self.ID_TO_TOKEN.get(token, '<UNK>')
            if token_str in self.SPECIAL_TOKENS:
                continue  # Skip special tokens
            chars.append(token_str)
        return ''.join(chars)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        tokens = tokens or self.encode(text)
        token_texts = []
        offsets = []
        idx = 0
        for token in tokens:
            token_str = self.ID_TO_TOKEN.get(token, '<UNK>')
            if token_str in self.SPECIAL_TOKENS:
                continue  # Skip special tokens
            token_texts.append(token_str)
            offsets.append(idx)
            idx += 1
        return token_texts, offsets


def build_tokenizer(name: str, path: Optional[str] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        assert has_sp, "sentencepiece not installed"
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        assert has_tiktoken, "tiktoken not installed"
        return TikTokenTokenizer(path)
    elif name == "aa":
        return AminoAcidTokenizer()
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
