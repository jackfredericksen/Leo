"""
Shared bot state — imported by both main.py and web_gui.py so they
always reference the same objects regardless of how the entry point
is invoked (__main__ vs main).
"""

# These are set by main.py at startup before any web requests are served.
state = None
config = None
_pos_manager_ref = None
_storage_ref = None
