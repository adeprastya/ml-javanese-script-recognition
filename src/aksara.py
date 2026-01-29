NGLEGENA = [
    ("ꦲ", "ha"),
    ("ꦤ", "na"),
    ("ꦕ", "ca"),
    ("ꦫ", "ra"),
    ("ꦏ", "ka"),
    ("ꦢ", "da"),
    ("ꦠ", "ta"),
    ("ꦱ", "sa"),
    ("ꦮ", "wa"),
    ("ꦭ", "la"),
    ("ꦥ", "pa"),
    ("ꦝ", "dha"),
    ("ꦗ", "ja"),
    ("ꦪ", "ya"),
    ("ꦚ", "nya"),
    ("ꦩ", "ma"),
    ("ꦒ", "ga"),
    ("ꦧ", "ba"),
    ("ꦛ", "tha"),
    ("ꦔ", "nga"),
]

CHAR_LIST = [c for c, _ in NGLEGENA]

BLANK_IDX = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHAR_LIST)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHAR_LIST)}

assert len(CHAR_LIST) == len(CHAR2IDX) == len(IDX2CHAR)
