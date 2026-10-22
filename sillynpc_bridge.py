"""Decode SillyNPC NPC fields carried through SillyTavern's fixed voice proxy."""
import base64
import binascii
import json
import re


def decode_sillynpc_voice(body):
    prefix = "sillynpc-v1:"
    if not body.voice or not body.voice.startswith(prefix):
        return body
    if body.npc_id or body.instructions:
        raise ValueError("SillyNPC voice metadata cannot be combined with other NPC fields")
    encoded = body.voice[len(prefix):]
    try:
        decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise ValueError("Invalid SillyNPC voice metadata") from exc
    if not isinstance(data, dict) or set(data) != {"npc_id", "instructions", "max_new_tokens"}:
        raise ValueError("Invalid SillyNPC voice metadata fields")
    if not isinstance(data["npc_id"], str) or not re.fullmatch(r"sn_[A-Za-z0-9_-]{1,125}", data["npc_id"]):
        raise ValueError("Invalid SillyNPC NPC ID")
    if not isinstance(data["instructions"], str) or not data["instructions"].strip():
        raise ValueError("SillyNPC voice description is required")
    if type(data["max_new_tokens"]) is not int or not 2 <= data["max_new_tokens"] <= 4096:
        raise ValueError("Invalid SillyNPC speech token budget")
    return type(body).model_validate({**body.model_dump(), "voice": "alloy", **data})
