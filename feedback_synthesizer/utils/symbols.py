"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
# from . import cmudict

_pad        = "_"
_eos        = "~"
_characters = '0123456789!\'\"(),-.:;?#@$%& '
phones = 'EH N J JH W IY DH HH Q IH T NG P X K TH F D CH V ZH UH EY AW AX B L Y SH S AH AY R ER AE UW G Z OW AO OY M AA'
#_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + phones.split() + list(_characters) #+ _arpabet
