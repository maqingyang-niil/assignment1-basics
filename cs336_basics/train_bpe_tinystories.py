from cs336_basics.bpe import train_bpe
import time
import pickle
import os


## 特殊token处理
special_tokens=[
    "<|endoftext|>"
]




def main():
    workers=15
    start=time.time()
    vocab,merges=train_bpe("../data/TinyStoriesV2-GPT4-train.txt",10000,special_tokens,workers)
    end=time.time()
    period=end-start
    print(period)
    os.makedirs("../log",exist_ok=True)
    with open("../log/vocab.pkl","wb") as f:
        pickle.dump(vocab,f)
    with open("../log/merges.pkl","wb") as f:
        pickle.dump(merges,f)


if __name__=="__main__":
    main()