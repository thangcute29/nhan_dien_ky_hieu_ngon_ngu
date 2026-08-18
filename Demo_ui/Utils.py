# Demo_ui/Utils.py
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Shared_lib.ui_helpers import TextToSpeech, VirtualCamera, SubtitleRenderer, draw_hand_badge
