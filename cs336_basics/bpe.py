import regex as re
from cs336_basics.pretokenization import find_chunk_boundaries
from concurrent.futures import ProcessPoolExecutor
import heapq
import os

class ReverseBytes:
    __slots__ = ("data",)
    def __init__(self, data):
        self.data = data
    def __lt__(self, other):
        return self.data > other.data  # 反转比较方向
    def __eq__(self, other):
        return self.data == other.data

def rebuild_heap(global_dict_pair):
    heap = []
    for pair, count in global_dict_pair.items():
        heapq.heappush(heap, (-count, ReverseBytes(pair[0]), ReverseBytes(pair[1]), pair))
    return heap

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

## 重建新堆
def rebuild_heap(global_dict_pair):
    heap = []
    for pair, count in global_dict_pair.items():
        heapq.heappush(heap, (-count, ReverseBytes(pair[0]), ReverseBytes(pair[1]), pair))
    return heap
## 统计chunk中的字符个数
def get_statistic(
        start:int,
        end:int,
        input_path:str,
        special_tokens:list[str]
)->dict[bytes,int]:
    word_dict = {}
    delimiter="|".join(
        re.escape(token)
        for token in special_tokens
    )
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk=f.read(end-start)
        chunk=chunk.decode("utf-8")

        segments=re.split(delimiter,chunk)

        for segment in segments:
            for match in re.finditer(PAT,segment):
                word=match.group().encode("utf-8")
                word_dict[word]=word_dict.get(word,0)+1

    return word_dict

def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        workers:int
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    eot_token=special_tokens[special_tokens.index("<|endoftext|>")]
    eot_bytes=eot_token.encode("utf-8")

## 获取文件分割点，以及chunk的数量
    max_workers=os.cpu_count()
    if workers>max_workers:
        workers=max_workers

    with open(input_path,"rb") as f:
        boundaries=find_chunk_boundaries(f,
                                         workers,
                                         eot_bytes)
        chunk_num=len(boundaries)-1

## 线程池处理全部的文本
    with ProcessPoolExecutor(max_workers=chunk_num) as executor:
        futures=[]
        for i in range(chunk_num):
            start=boundaries[i]
            end=boundaries[i+1]
            future=executor.submit(
                get_statistic,
                start,
                end,
                input_path,
                special_tokens,
            )
            futures.append(future)

        local_dict=[]
        for future in futures:
            local_dict.append(future.result())
## 得到整体的字符统计
    global_dict={}
    for local in local_dict:
        for word,count in local.items():
            global_dict[word]=global_dict.get(word,0)+count

## 将单词转化为字母元组
    global_dict_tuple={}
    for k,v in global_dict.items():
        global_dict_tuple[tuple(bytes([c]) for c in k)]=v

## 单词中的字母两两组合
## 以及这个两两组合出现在哪些单词元组中
    global_dict_pair={}
    global_pair_to_word={}

    for word,freq in global_dict_tuple.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            global_dict_pair[pair]=global_dict_pair.get(pair,0)+freq

            global_pair_to_word.setdefault(pair,set()).add(word)

            count=global_dict_pair[pair]
    
    heap = []
    for pair, count in global_dict_pair.items():
        heapq.heappush(heap, (-count, ReverseBytes(pair[0]), ReverseBytes(pair[1]), pair))

## 初始化vocab
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    for i in range(len(special_tokens)):
        vocab[256+i]=special_tokens[i].encode("utf-8")

## merge对应的pair
    num_merge=vocab_size-256-len(special_tokens)
    merges=[]

    for i in range(num_merge):
        if len(heap)>2*len(global_dict_pair):
            heap=rebuild_heap(global_dict_pair)

        while True:
            neg_count,_,_,pair=heapq.heappop(heap)
            count=-neg_count
            if global_dict_pair.get(pair,0)==count:
                max_pair=pair
                break
        
        merges.append(max_pair)
        new_token=max_pair[0]+max_pair[1]
        vocab[len(vocab)]=new_token

## 合并max_pair
        for word in global_pair_to_word[max_pair].copy():
            freq=global_dict_tuple[word]
            new_word=[]
            i=0
## 提取新word
            while i<len(word):
                if i+1<len(word) and (word[i],word[i+1])==max_pair:
                    new_word.append(max_pair[0]+max_pair[1])
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
                    heapq.heappush(heap,(-new_count,ReverseBytes(pair[0]),ReverseBytes(pair[1]),pair))
## 加上新pair的频率
            for pair,count in new_pairs.items():
                new_count=global_dict_pair.get(pair,0)+count*freq
                global_dict_pair[pair]=new_count
                heapq.heappush(heap,(-new_count,ReverseBytes(pair[0]),ReverseBytes(pair[1]),pair))
## 删掉旧word
            del global_dict_tuple[word]
## 增加新word
            global_dict_tuple[new_word]=global_dict_tuple.get(new_word,0)+freq
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
