import regex as re
from cs336_basics.pretokenization import find_chunk_boundaries
from concurrent.futures import ProcessPoolExecutor,as_completed
import heapq
import os
from collections import Counter

class ReverseBytes:
    __slots__ = ("id", "vocab")
    def __init__(self, id_, vocab):
        self.id = id_
        self.vocab = vocab
    def __lt__(self, other):
        return self.vocab[self.id] > self.vocab[other.id]
    def __eq__(self, other):
        return self.id == other.id

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

## 重建新堆
def rebuild_heap(global_dict_pair,vocab):
    heap = []
    for pair, count in global_dict_pair.items():
        heapq.heappush(heap, (-count, ReverseBytes(pair[0],vocab), ReverseBytes(pair[1],vocab), pair))
    return heap
## 统计chunk中的字符个数
def get_statistic(
        start:int,
        end:int,
        input_path:str,
        special_tokens:list[str]
)->dict[tuple[int,...],int]:
    word_dict = {}
    delimiter="|".join(re.escape(token) for token in special_tokens)
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk=f.read(end-start).decode("utf-8")

        segments=re.split(delimiter,chunk)
        for segment in segments:
            for match in re.finditer(PAT,segment):
                word_bytes=match.group().encode("utf-8")
                word_tuple=tuple(word_bytes)
                word_dict[word_tuple]=word_dict.get(word_tuple,0)+1
    return word_dict

def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        workers:int,
        chunks:int
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    eot_token=special_tokens[special_tokens.index("<|endoftext|>")]
    eot_bytes=eot_token.encode("utf-8")

## 获取文件分割点，以及chunk的数量
    max_workers=os.cpu_count()
    if workers>max_workers:
        workers=max_workers

    with open(input_path,"rb") as f:
        boundaries=find_chunk_boundaries(f,
                                         chunks,
                                         eot_bytes)
        chunk_num=len(boundaries)-1
    print("Finish splitting documents into chunks")
## 多进程处理全部的文本
    global_dict=Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(get_statistic,boundaries[i],boundaries[i+1],input_path,special_tokens)
                 for i in range(chunk_num)]
        for future in as_completed(futures):
            global_dict.update(future.result())
    print("Finish processing chunks in multithreads")
## 单词中的字母两两组合
## 以及这个两两组合出现在哪些单词元组中
    global_dict_pair={}
    global_pair_to_word={}

    for word,freq in global_dict.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            global_dict_pair[pair]=global_dict_pair.get(pair,0)+freq

            global_pair_to_word.setdefault(pair,set()).add(word)

## 初始化vocab
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    for i in range(len(special_tokens)):
        vocab[256+i]=special_tokens[i].encode("utf-8")
    
    
    heap = []
    for pair, count in global_dict_pair.items():
        heapq.heappush(heap, (-count, ReverseBytes(pair[0],vocab), ReverseBytes(pair[1],vocab), pair))

## merge对应的pair
    num_merge=vocab_size-256-len(special_tokens)
    merges=[]
    print("Finish initialising basic data")
    print("Start to merge")
    for i in range(num_merge):
        if len(heap)>2*len(global_dict_pair):
            heap=rebuild_heap(global_dict_pair,vocab)

        while True:
            neg_count,_,_,pair=heapq.heappop(heap)
            count=-neg_count
            if global_dict_pair.get(pair,0)==count:
                max_pair=pair
                break

        new_id=len(vocab)
        merges.append((vocab[max_pair[0]],vocab[max_pair[1]]))
        vocab[new_id]=vocab[max_pair[0]]+vocab[max_pair[1]]

## 合并max_pair
        for word in global_pair_to_word[max_pair].copy():
            freq=global_dict[word]
            new_word=[]
            i=0
## 提取新word
            while i<len(word):
                if i+1<len(word) and (word[i],word[i+1])==max_pair:
                    new_word.append(new_id)
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_word=tuple(new_word)
## 提取旧pair
            old_pairs={}
            for i in range(len(word)-1):
                pair=(word[i],word[i+1])
                old_pairs[pair]=old_pairs.get(pair,0)+1
## 提取新pair
            new_pairs={}
            for i in range(len(new_word)-1):
                pair=(new_word[i],new_word[i+1])
                new_pairs[pair]=new_pairs.get(pair,0)+1
## 减去旧pair的频率
            for pair,count in old_pairs.items():
                new_count=global_dict_pair[pair]-count*freq
                if new_count<=0:
                    del global_dict_pair[pair]
                else:
                    global_dict_pair[pair]=new_count
                    heapq.heappush(heap,(-new_count,ReverseBytes(pair[0],vocab),ReverseBytes(pair[1],vocab),pair))
## 加上新pair的频率
            for pair,count in new_pairs.items():
                new_count=global_dict_pair.get(pair,0)+count*freq
                global_dict_pair[pair]=new_count
                heapq.heappush(heap,(-new_count,ReverseBytes(pair[0],vocab),ReverseBytes(pair[1],vocab),pair))
## 删掉旧word
            del global_dict[word]
## 增加新word
            global_dict[new_word]=global_dict.get(new_word,0)+freq
## 删掉旧pair对旧word的所有索引
            for pair in old_pairs:
                if pair in global_pair_to_word:
                    global_pair_to_word[pair].discard(word)

                    if not global_pair_to_word[pair]:
                        del global_pair_to_word[pair]
## 增加新pair对新word的所有索引
            for pair in new_pairs:
                global_pair_to_word.setdefault(pair,set()).add(new_word)
            

    return vocab,merges
