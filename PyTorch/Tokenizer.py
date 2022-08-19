import torch
from transformers import BertTokenizer
import emoji
import tqdm


REMOVE_PATTERNS = {'\ufe0f': '',
                   '\u2019': "'",
                   '\u2026': '...',
                   '\u2654': "<3"}

def get_pretrained_tokenizer():
    """
    Loads pretrained BERT Tokenizer and returns it,

    """
    print("Downloading bert tokenizer to cache")
    print("---------------------------------------")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                   do_lower_case=True
                                                   )
    return bert_tokenizer


def handle_emoji_codes(string):
    string = repr(string)
    # for p, n in REMOVE_PATTERNS.items():
    #     string = string.replace(p, n)
    weirdInput = string.encode('ascii', 'replace').decode().encode('latin-1')
    output = (weirdInput
              .decode("raw_unicode_escape")
              .encode('utf-16', 'surrogatepass')
              .decode('utf-16')
              .encode("raw_unicode_escape")
              .decode("latin_1")
              )
    emojized = emoji.emojize(output.encode().decode('unicode_escape'))
    emojized = emojized.replace('\n', '')
    return emojized


def decode(ids: torch.Tensor,
           tokenizer: BertTokenizer) -> str:
    return tokenizer.decode(ids.tolist()[0])



def tokenize_sentences(bert_tokenizer, sentences, aspects, maxlen):
    """ converts sentences into ids according to bert tokenizer

    Arguments:
    bert_tokenizer (Tokenizer): Pretrained BERT Tokenizer
    sentences (list): List of sentences
    aspects (list): List of Aspects
    maxlen (int): Maximum Length of a sentence

    """
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for sentence, aspect in zip(sentences, aspects):
        sentence = handle_emoji_codes(sentence)

        encoded = bert_tokenizer.encode_plus(text=sentence,
                                             text_pair=aspect,
                                             add_special_tokens=True,
                                             max_length=maxlen,
                                             truncation=True,
                                             padding='max_length',
                                             return_token_type_ids=True,
                                             return_tensors='pt')

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0, out=None)
    attention_masks = torch.cat(attention_masks, dim=0, out=None)
    token_type_ids = torch.cat(token_type_ids, dim=0, out=None)

    return input_ids, attention_masks, token_type_ids
