# bpe_tokenizer.py（最终修正版）
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import argparse
from typing import List, Optional

class BPETokenizer:
    def __init__(self, vocab_size=5000, special_tokens=None):  # 恢复参数名称
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = special_tokens or ["[UNK]", "[PAD]"]
        self._target_vocab_size = vocab_size  # 使用私有变量避免冲突

        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Digits(individual_digits=True)
        ])

    def train(self, corpus_path, output_path="bpe_tokenizer.json"):
        """训练并保存分词器"""
        trainer = trainers.BpeTrainer(
            vocab_size=self._target_vocab_size,  # 使用私有变量
            special_tokens=self.special_tokens,
            min_frequency=2
        )

        self.tokenizer.train(
            files=[corpus_path],
            trainer=trainer
        )
        self.tokenizer.save(output_path)
        print(f"BPE分词器已保存至 {output_path}")

    @classmethod
    def load(cls, model_path="bpe_tokenizer.json"):
        """加载预训练分词器"""
        tokenizer = cls()  # 使用默认参数初始化
        tokenizer.tokenizer = Tokenizer.from_file(model_path)
        return tokenizer

    @property
    def vocab_size(self):
        """动态获取实际词汇量"""
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:  # 正确写法
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:  # 同步修正其他方法
        return self.tokenizer.decode(ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE分词器训练")
    parser.add_argument("--corpus", type=str, required=True, help="训练语料路径")
    parser.add_argument("--output", type=str, default="bpe_tokenizer.json", help="模型输出路径")
    parser.add_argument("--vocab-size", type=int, default=5000, help="词汇表大小")
    args = parser.parse_args()

    # 修正参数传递名称
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(args.corpus, args.output)
    print(f"实际词汇量: {tokenizer.vocab_size}")
    tokenizer = BPETokenizer.load("bpe_tokenizer.json")
    sample_text = "自然语言处理"
    print(tokenizer.encode(sample_text))  # 应输出类似[235, 1780, 432]