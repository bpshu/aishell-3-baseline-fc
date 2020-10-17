from .symbols import symbols
from . import cleaners
import re

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
tone_re = re.compile(r'([0-9]+)')
phone_re = re.compile(r'([^0-9]+)')
lg_tokens_ids = [0, 1, 2]

def text_to_sequence(text, cleaner_names):
  """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  """

  lg_seq = ' '.join(['1'] * len(text.split()))

  assert len(text.split()) == len(lg_seq.split()) 
  sequence = []
  language_tokens = []
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      text_seq, language_seq = _symbols_to_sequence(_clean_text(text, cleaner_names), lg_seq)
      sequence += text_seq
      language_tokens += language_seq
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  # Append EOS token
  sequence.append(_symbol_to_id["~"])
  language_tokens.append(2)
  return sequence


def sequence_to_text(sequence):
  """Converts a sequence of IDs back to a string"""
  result = ""
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == "@":
        s = "{%s}" % s[1:]
      result += s
  return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception("Unknown cleaner: %s" % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols, lg_seq):
  sequence = []
  lg_sequence = []
  lg_tokens = lg_seq.split()
  phone_tokens = symbols.split()
  for i in range(len(phone_tokens)):
    sym = phone_tokens[i]
    if len(tone_re.findall(sym)) > 0:
      phone = phone_re.findall(sym)[0]
      tone = tone_re.findall(sym)[0]
      if phone not in _symbol_to_id.keys():
        continue
      sequence.append(_symbol_to_id[phone])
      sequence.append(_symbol_to_id[tone])
      lgt = int(lg_tokens[i])
      assert lgt in lg_tokens_ids
      lg_sequence.append(lgt)
      lg_sequence.append(lgt)
    else:
      if sym not in _symbol_to_id.keys():
        continue
      sequence.append(_symbol_to_id[sym])
      lgt = int(lg_tokens[i])
      assert lgt in lg_tokens_ids
      lg_sequence.append(lgt)
    sequence.append(_symbol_to_id[' ']) 
    lg_sequence.append(2)
        
  assert len(sequence) == len(lg_sequence)
  return sequence[:-1], lg_sequence[:-1]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s not in ("_", "~")

