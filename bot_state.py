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
_resume_event = None        # asyncio.Event — cleared when paused, set when running
_stop_event = None          # asyncio.Event — set to trigger graceful shutdown
_kyle_lambda_ref = None     # KyleLambdaTracker
_evolution_ref = None       # EvolutionAgent
_confluence_ref = None      # ConfluenceTracker
_hurst_ref = None           # HurstTracker
_trader_ref = None          # Trader — for cooldown management
_force_market_refresh = False   # set True to wake market_refresh_loop early
paused = False
