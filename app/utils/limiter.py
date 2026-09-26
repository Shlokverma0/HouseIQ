"""
limiter.py
----------
Rate limiter instance (kept separate to avoid circular imports).
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)