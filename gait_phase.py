from enum import Enum

class GaitPhase(Enum):
    INITIAL_CONTACT = 1
    LOADING_RESPONSE = 2
    MID_STANCE = 3
    TERMINAL_STANCE = 4
    PRE_SWING = 5
    INITIAL_SWING = 6
    MID_SWING = 7
    TERMINAL_SWING = 8