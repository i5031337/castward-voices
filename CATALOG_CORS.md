# Castward Speech catalog access

Castward's Speech tab reads models, voices, and voice-design capabilities from this server's `GET /v1/models`, `GET /v1/voices`, and `GET /ui/config` routes. The browser contacts the selected TTS endpoint directly.

For local SillyTavern at `http://127.0.0.1:8000` or `http://localhost:8000`, this server allows those origins through CORS. Open the Speech tab on the host machine to discover choices and save them in Castward's extension settings.

Remote browsers cannot refresh this catalog when the endpoint is `http://127.0.0.1:8001/v1/audio/speech`: `127.0.0.1` refers to the remote device, and a Tailscale browser origin is not in the CORS allowance. The saved model and voice choices still appear in Castward's dropdowns remotely. Speech generation uses SillyTavern's built-in TTS proxy.
