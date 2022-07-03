import enum

class hate_speech_type(enum.Enum):
    none = -1
    non_hate_speech = 0
    general_hate_speech = 1
    racism = 2
    sexism = 3
    islamophobia = 4