
class Vocab:
    def __init__(self, embed, word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'

    def __len__(self):
        return len(self.word2id)

    def i2w(self, idx):
        return self.id2word[idx]

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def get_ids(self, text):
        """Get ids corresponding to words in text.
    Assumes tokens separated by space.

    Args:
      text: a string with tokens separated by space.

    Returns:
      A list of ints representing word ids.
    """
        return [self.w2i(w) for w in text.split()]

