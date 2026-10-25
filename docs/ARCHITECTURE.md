# Architecture

## Audio path

1. The browser decodes a selected file through a 44.1 kHz AudioContext.
2. Stereo float PCM streams to the server over a WebSocket.
3. Each listener collects audio into overlapping windows.
4. A shared worker runs query-conditioned SCNet inference, batching requests where possible.
5. Overlap-add blends predictions; instrument changes use a cross-fade.
6. The server returns the selected stem and residual in a numbered binary frame.
7. The browser applies listening-mode gains and schedules audio.
8. A bundled SoundTouchJS AudioWorklet transposes the audible mix by −12 to +12 semitones, keeping source playback rate at 1.

The returned frame header is little-endian `<IqII`: sequence number, starting sample, sample count, and instrument index. Stereo stem samples precede stereo residual samples. See `scripts/stream_server.py` and `web/index.html` for the protocol.

## Model

SCNet predicts a complex target spectrogram. FiLM layers use an instrument embedding to scale and shift internal features. The learned queries are vocals, drums, guitar, and piano. “Other” is calculated by subtraction, not learned as a fifth query.

The default browser demo offers vocals, drums, piano, and everything else. Its residual includes guitar. Offline five-stem evaluation instead removes all four learned queries before computing other.

## Frontend

A single HTML document contains the homepage (`/`) and studio (`/#studio`). The server serves this file directly; no Node build step is required. The SoundTouchJS processor and source map are bundled in an inert script block and loaded into an AudioWorklet via a Blob URL. The processor license and source attribution are retained.

Google Fonts provide Syne, Space Grotesk, and Bodoni Moda, with local fallback fonts. Pitch requires localhost or HTTPS and a browser supporting AudioWorklet. Unsupported browsers can still use unshifted playback.

## Existing remote demo

`remote/demo.sh` is an optional helper for the original SSH-hosted setup. It expects the project at `~/audio_csesa` on the remote machine, an existing Python environment, checkpoint, and demo assets. Set `HOST` to an SSH alias if needed. It forwards ports 8765 and 8766. It is not a general deployment installer.
