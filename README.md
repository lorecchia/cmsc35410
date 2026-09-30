# CMSC 35410 — Spectral Methods for Machine Learning and Network Analysis

Course website and materials. The site is built with [Quarto](https://quarto.org) and published to
**https://lorecchia.github.io/cmsc35410** by GitHub Actions on every push to `main`.

## Layout

```
_quarto.yml            site config (navbar, theme)
index.qmd              home page: description, logistics, topics
schedule.qmd           lecture-by-lecture schedule with links
notes.qmd              auto-generated list of everything in notes/
homework.qmd           problem sets
resources.qmd          references, software
lectures/lectureNN/    one folder per lecture: reveal.js deck (.qmd), theme, demos/
notes/                 lecture notes and supplements (.qmd)
homework/              problem sets (no solutions — this repo is public)
exercises/             recommended exercises, linked from schedule.qmd (solutions kept locally, git-ignored)
assets/site.scss       site colours and fonts
_freeze/               stored results of Python cells (commit this)
```

## Editing workflow

```bash
brew install quarto
pip install -r requirements.txt
quarto preview            # live preview of the whole site at localhost
```

Then commit and push; the site updates in a minute or two.

- **New lecture:** copy `lectures/lecture01/` to `lectures/lecture02/`, edit the `.qmd`, add a row to `schedule.qmd`.
- **New note:** drop a `.qmd` with `title`/`subtitle`/`description` front matter into `notes/`; it appears on the Notes page automatically.
- **Python cells:** results are frozen. After editing a deck with code, run `quarto render <file>` locally and commit the updated `_freeze/` folder.
- **Solutions and grades** stay in Box, not here (`.gitignore` blocks common names as a safety net).
