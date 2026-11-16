# handwriting-animator

Animates text as if a pen were writing it, stroke by stroke, from any TrueType or OpenType font.

## How it works

- **Shaping.** HarfBuzz lays out the text so ligatures, kerning and glyph positions match what the font intends.
- **Outlines to strokes.** fontTools reads each glyph's outline. A custom pen flattens the Bézier curves with adaptive subdivision: tight curves get more points, straight runs get fewer.
- **Stroke order.** Contours are ordered outside-to-inside and top-to-bottom, close to how a person would draw the letter.
- **Animation.** matplotlib reveals the strokes at constant pen speed, with Replay and Exit buttons.

## Run it

```bash
pip install -r requirements.txt
./run.sh
```

Pick a font from the list. The text, size, duration and frame rate are constants at the top of `animate_text.py` (`TEXT`, `FONT_SIZE`, `DURATION`, `FPS`). Set `DEBUG = True` to print stroke statistics.

## Fonts

`fonts/` ships five script fonts from Google Fonts under the SIL Open Font License: Alex Brush, Cookie, Explora, Parisienne and Tangerine. Drop any other `.ttf` or `.otf` file in that folder and it shows up in the picker.
