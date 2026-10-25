# Demo tracks

The repository contains an empty `index.json` so a fresh checkout does not list missing audio files. You can always use **Open audio** in Flora once the server has a model checkpoint.

To add demo tracks:

1. Place audio you are permitted to use in this directory.
2. Edit `index.json`, using `index.example.json` as the format guide.
3. Set each `file` to a local filename and `title` to its display title.
4. Refresh the studio.

Audio files are excluded by `.gitignore`. The existing university demo catalog was preserved in the local archive; this repository cleanup does not change the running server.
