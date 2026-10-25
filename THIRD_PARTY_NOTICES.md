# Third-party model attribution

## PocketTTS 100M

Pocket TTS is created by **Kyutai**. Original model and model card:
<https://huggingface.co/kyutai/pocket-tts>.

The model weights are licensed under **Creative Commons Attribution 4.0 International
(CC-BY-4.0)**: <https://creativecommons.org/licenses/by/4.0/>.
License text: <https://creativecommons.org/licenses/by/4.0/legalcode.en>.

This integration uses the English Q8_0 GGUF conversion and packaged voice states
from [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/ef1ab2598f6d4d101486e4c5b9e605c687f20642/PocketTTS-GGUF/english),
revision `ef1ab2598f6d4d101486e4c5b9e605c687f20642`.
The conversion changes the weight representation to GGUF/Q8_0. Castward does not
modify or fine-tune these downloaded weights or packaged voice states.
The repository's software license does not replace the model license.
