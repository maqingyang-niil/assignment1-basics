from cs336_basics.tokenizer import tokenizer
import random


special_tokens=[
    "<|endoftext|>"
]

## 水塘抽样法读取
def sample_documents(filepath: str, n: int = 10, seed: int = 42, delimiter: str = "<|endoftext|>") -> list[str]:
    random.seed(seed)
    reservoir = []
    count = 0
    buffer = ""

    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 每次只读1MB
            if not chunk:
                break
            buffer += chunk
            # 尽量切出完整的文档，保留最后一段可能不完整的部分到下次
            parts = buffer.split(delimiter)
            buffer = parts[-1]          # 最后一段可能被截断，留到下一轮
            for doc in parts[:-1]:
                doc = doc.strip()
                if not doc:
                    continue
                count += 1
                if len(reservoir) < n:
                    reservoir.append(doc)
                else:
                    j = random.randint(0, count - 1)
                    if j < n:
                        reservoir[j] = doc

        # 处理文件末尾剩下的最后一段
        doc = buffer.strip()
        if doc:
            count += 1
            if len(reservoir) < n:
                reservoir.append(doc)
            else:
                j = random.randint(0, count - 1)
                if j < n:
                    reservoir[j] = doc

    return reservoir

tinystories_tokenizer=tokenizer.from_files(
    vocab_filepath="../log/vocab_ts.pkl",
    merges_filepath="../log/merges_ts.pkl",
    special_tokens=["<|endoftext|>"]
)

openwebtext_tokenizer=tokenizer.from_files(
    vocab_filepath="../log/vocab_owt.pkl",
    merges_filepath="../log/merges_owt.pkl",
    special_tokens=["<|endoftext|>"]
)

def compression_ratio(tokenizer,documents:list[str])->float:
    total_bytes=0
    total_tokens=0
    for doc in documents:
        doc_bytes=len(doc.encode("utf-8"))
        ids=tokenizer.encode(doc)
        total_bytes+=doc_bytes
        total_tokens+=len(ids)
    return total_bytes/total_tokens

ts_docs=sample_documents("../data/TinyStoriesV2-GPT4-train.txt",n=10)
owt_docs=sample_documents("/media/cronusiius/Data/datasets/openwebtext/owt_train_3G.txt",n=10)

ts_ratio=compression_ratio(tinystories_tokenizer,ts_docs)
owt_ratio=compression_ratio(openwebtext_tokenizer,owt_docs)


print(f"TinyStories tokenizer on TinyStories docs: {ts_ratio:.3f} bytes/token")
print(f"OpenWebText tokenizer on OpenWebText docs: {owt_ratio:.3f} bytes/token")