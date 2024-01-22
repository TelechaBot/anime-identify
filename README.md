# AnimeIDF

Toolkit for Distinguishing Anime Images from Real Images With Lightweight Dependency.

[![PyPI](https://img.shields.io/pypi/v/anime-identify.svg)](https://pypi.org/project/anime-identify/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/anime-identify.svg)](https://pypi.org/project/anime-identify/)

## Usage

```shell
pip install anime-identify
```

```python
from anime_identify import AnimeIDF


def test_anime():
    with open("anime.jpg", "rb") as f:
        f.seek(0)
        poss = (AnimeIDF().predict_image(content=f))
    print(poss)
    assert poss > 50, f"anime poss: {poss}"


def test_real():
    poss = AnimeIDF().predict_image("real.jpg")
    print(poss)
    assert poss < 50, f"real poss: {poss}"

```