import pickle
import regex as re
from typing import Iterable,Iterator



PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class tokenizer:
## 初始化
    def __init__(self,vocab:dict[int,bytes],
                 merges:list[tuple[bytes,bytes]],
                 special_tokens:list[str]|None=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens or []
        self.token_to_id={val:key for key,val in vocab.items()}
        self.merge_ranks={pair: i for i,pair in enumerate(merges)}
## 按照merges合并单词中的pair
    def _merge_word(self,word_tuple:tuple[bytes,...])->tuple[bytes,...]:
        word=list(word_tuple)
        while len(word)>1:
            pairs=[(word[i],word[i+1]) for i in range(len(word)-1)]
            best_pair=min(pairs,key=lambda p: self.merge_ranks.get(p,float("inf")))
            if best_pair not in self.merge_ranks:
                break

            new_word=[]
            i=0
            while i<len(word):
                if i+1<len(word) and (word[i],word[i+1])==best_pair:
                    new_word.append(word[i]+word[i+1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            word=new_word
        return tuple(word)
## 从文件中读取vocab和merges
    @classmethod
    def from_files(cls,vocab_filepath:str,
                   merges_filepath:str,
                   special_tokens:list[str]|None=None):
        with open(vocab_filepath,"rb") as f:
            vocab=pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges=pickle.load(f)

        return cls(vocab,merges,special_tokens)
## encode函数实现
    def encode(self,text:str)->list[int]:
        word_id=[]
        if self.special_tokens:
            sorted_tokens=sorted(self.special_tokens,key=len,reverse=True)
            delimiter="|".join(re.escape(token) for token in sorted_tokens)
            segments=re.split(f"({delimiter})",text)## 保留分隔符
        else:
            segments=[text]

        for segment in segments:
            if not segment:
                continue
            if segment in self.special_tokens:
                special_bytes=segment.encode("utf-8")
                word_id.append(self.token_to_id[special_bytes])
                continue
            for match in re.finditer(PAT,segment):
                word_bytes=match.group().encode("utf-8")
                word_tuple=tuple(bytes([c]) for c in word_bytes)
                merged=self._merge_word(word_tuple)
                for token in merged:
                    word_id.append(self.token_to_id[token])
        return word_id
## 流式encode函数实现
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
## decode函数实现
    def decode(self,ids:list[int])->str:
        text=b"".join(self.vocab[i] for i in ids).decode("utf-8",errors="replace")
        return text
    
