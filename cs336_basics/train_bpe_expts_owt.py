from cs336_basics.bpe import train_bpe
import os
import time
import pickle



special_tokens=[
    "<|endoftext|>"
]

def main():
    workers=8
    start=time.time()
    chunks=64
    vocab,merges=train_bpe("/media/cronusiius/Data/datasets/openwebtext/openwebtext_500M.txt",32000,special_tokens,workers,chunks)
    print(time.time()-start)

    os.makedirs("../log",exist_ok=True)

    with open("../log/vocab_owt.pkl","wb") as f:
        pickle.dump(vocab,f)
    with open("../log/merges_owt.pkl","wb") as f:
        pickle.dump(merges,f)

if __name__=="__main__":
    main()