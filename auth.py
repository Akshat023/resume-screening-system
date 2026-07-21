"""
auth.py — Flask auth middleware
Next.js handles login/signup directly via Supabase Auth SDK.
Flask only verifies the JWT and enforces daily limits.
"""

import os
import json
import urllib.request
import urllib.error
from functools import wraps
from flask import request, jsonify

SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
DAILY_LIMIT       = int(os.getenv("DAILY_LIMIT", "5"))


def verify_token(token: str) -> dict | None:
    """
    Verify Supabase JWT by calling /auth/v1/user.
    Returns user dict if valid, None if invalid/expired.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        req = urllib.request.Request(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey":        SUPABASE_ANON_KEY,
                "Content-Type":  "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            return json.loads(res.read().decode())
    except Exception:
        return None


def require_auth(f):
    """
    Decorator — verifies Bearer token.
    Injects current_user into function kwargs.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authentication required"}), 401
        user = verify_token(auth_header.split(" ", 1)[1])
        if not user:
            return jsonify({"error": "Invalid or expired session. Please log in again."}), 401
        return f(*args, current_user=user, **kwargs)
    return decorated


def require_auth_with_limit(db):
    """
    Decorator factory — verifies token + enforces daily screening limit.

    Usage:
        @app.route("/api/screen", methods=["POST"])
        @require_auth_with_limit(db)
        def screen(current_user):
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Authentication required"}), 401

            user = verify_token(auth_header.split(" ", 1)[1])
            if not user:
                return jsonify({"error": "Invalid or expired session. Please log in again."}), 401

            user_id    = user["id"]
            used_today = db.get_usage_today(user_id, "resume_screening")

            if used_today >= DAILY_LIMIT:
                return jsonify({
                    "error":   "daily_limit_reached",
                    "message": f"Daily limit reached. You have used {used_today}/{DAILY_LIMIT} screenings today. Resets at midnight UTC.",
                    "used":    used_today,
                    "limit":   DAILY_LIMIT,
                }), 429

            return f(*args, current_user=user, **kwargs)
        return decorated
    return decorator
