# Why there is no `packages.txt`

Streamlit Community Cloud runs `apt-get` on every build if — and only if — the
repository contains a `packages.txt`. This one used to contain a single line,
`ffmpeg`, and that line was enough to make deployment fail entirely:

```
E: Release file for http://deb.debian.org/debian-security/dists/bullseye-security/InRelease
   is expired (invalid since 1h 52min 44s).
❗️ installer returned a non-zero exit code
❗️ Error during processing dependencies!
```

Nothing in this app was wrong. Debian's `bullseye-security` metadata expired on
the platform's base image, `apt-get update` returned non-zero, and the build
stopped before Python was ever reached. It is not a failure anyone deploying an
app can fix, and it recurs whenever that metadata lapses again.

**The system `ffmpeg` package was never needed.** Nothing in this codebase shells
out to `ffmpeg` or `ffprobe`. Audio decoding goes through `av` (PyAV), a direct
dependency in `requirements.txt`, and its manylinux wheels ship FFmpeg compiled
in — `libavcodec`, `libavformat`, `libavutil`, `libswresample` and the rest are
inside the wheel. faster-whisper decodes through the same library.

So `packages.txt` was removed. With no apt manifest, Community Cloud skips the
apt step, an expired Debian repository cannot break the build, and deploys are
faster besides.

## If you run this outside Streamlit Cloud

You still do not need system ffmpeg — `pip install -r requirements.txt` brings a
working decoder. Install it anyway if you want the command-line tools for
splitting recordings by hand:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install ffmpeg        # Windows
```

That is a convenience for you, not a requirement of the app.

## If a build fails on apt again

Do not add `packages.txt` back to install something unless you are certain it is
needed. A pure-Python dependency with binary wheels is always the safer choice on
a platform whose base image you do not control: it cannot be broken by someone
else's repository metadata.
