import regex as re
from cs336_basics.pretokenization import find_chunk_boundaries
from concurrent.futures import ThreadPoolExecutor

## 特殊token处理
special_token=[
    "<|endoftext|>"
]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
        special_tokens:list[str]
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    eot_token=special_tokens[special_tokens.index("<|endoftext|>")]
    eot_bytes=eot_token.encode("utf-8")

## 获取文件分割点，以及chunk的数量

    with open(input_path,"rb") as f:
        boundaries=find_chunk_boundaries(f,
                                         12,
                                         eot_bytes)
        thread_len=len(boundaries)-1

## 线程池处理全部的文本
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures=[]
        for i in range(thread_len):
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
## 获取频率最大的pair
    global_dict_pair={}
    global_pair_to_word={}

    max_pair=None
    max_count=0


    for word,freq in global_dict_tuple.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            global_dict_pair[pair]=global_dict_pair.get(pair,0)+freq

            global_pair_to_word.setdefault(pair,set()).add(word)

            count=global_dict_pair[pair]
            if (count>max_count or (count==max_count and (max_pair is None or pair>max_pair))):
                max_count=count
                max_pair=pair
    


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
        max_pair = max(global_dict_pair,key=lambda pair: (global_dict_pair[pair], pair))
        merges.append(max_pair)
        new_token=max_pair[0]+max_pair[1]
        vocab[len(vocab)]=new_token

## 合并max_pair
        for word in global_pair_to_word[max_pair].copy():
            freq=global_dict_tuple[word]
            new_word=[]
            i=0
            while i<len(word):
                if i+1<len(word) and (word[i],word[i+1])==max_pair:
                    new_word.append(max_pair[0]+max_pair[1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_word=tuple(new_word)

            old_pairs={}

            for i in range(len(word)-1):
                pair=(word[i],word[i+1])
                old_pairs[pair]=old_pairs.get(pair,0)+1

            new_pairs={}

            for i in range(len(new_word)-1):
                pair=(new_word[i],new_word[i+1])
                new_pairs[pair]=new_pairs.get(pair,0)+1

            for pair,count in old_pairs.items():
                global_dict_pair[pair]-=count*freq
                if global_dict_pair[pair]==0:
                    del global_dict_pair[pair]

            for pair,count in new_pairs.items():
                global_dict_pair[pair]=global_dict_pair.get(pair,0)+count*freq

            del global_dict_tuple[word]

            global_dict_tuple[new_word]=global_dict_tuple.get(new_word,0)+freq

            for pair in old_pairs:
                if pair in global_pair_to_word:
                    global_pair_to_word[pair].discard(word)

                    if not global_pair_to_word[pair]:
                        del global_pair_to_word[pair]

            for pair in new_pairs:
                global_pair_to_word.setdefault(pair,set()).add(new_word)
            

    return vocab,merges
