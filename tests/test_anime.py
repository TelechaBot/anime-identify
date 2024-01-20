# -*- coding: utf-8 -*-
# @Time    : 2024/1/21 上午1:45
# @Author  : sudoskys
# @File    : test_anime.py
# @Software: PyCharm
from anime_identify import AnimeIDF


def test_anime():
    with open("anime.jpg", "rb") as f:
        poss = (AnimeIDF().predict_image(content=f))
    print(poss)
    assert poss > 50, f"anime poss: {poss}"


def test_real():
    poss = AnimeIDF().predict_image("real.jpg")
    print(poss)
    assert poss < 50, f"real poss: {poss}"
