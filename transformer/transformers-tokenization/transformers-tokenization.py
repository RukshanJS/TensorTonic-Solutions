import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {
            self.pad_token:0,
            self.unk_token:1,
            self.bos_token:2,
            self.eos_token:3,
        }
        self.id_to_word = {
            0:self.pad_token,
            1:self.unk_token,
            2:self.bos_token,
            3:self.eos_token,
        }

        unique_words=set()


        for text in texts:
            words=text.lower().split()
            unique_words.update(words)

        sorted_words=sorted(unique_words)

        current_id=4

        for word in sorted_words:
            self.word_to_id[word]=current_id
            self.id_to_word[current_id]=word
            current_id+=1

        self.vocab_size=len(self.word_to_id)

        return self.word_to_id

        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE

        words=text.lower().split()
        ids=[]
        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])

        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        words=[]
        for i in ids:
            words.append(self.id_to_word[i])

        return " ".join(words)