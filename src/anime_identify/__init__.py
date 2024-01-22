import os
import tempfile
from io import TextIOBase
from pathlib import Path
from typing import Union, IO

import cv2
import numpy as np
import onnxruntime
from robust_downloader import download


class AnimeIDF:
    _instance = None
    model_name = "anime_idf.onnx"  # 模型保存的路径
    # 家目录的 .cache
    model_folder = Path().home().joinpath(".cache", "anime_idf")
    model_url = "https://huggingface.co/mew233/AnimeIDF/resolve/main/anime_idf.onnx?download=true"

    @property
    def model_path(self):
        return os.path.join(self.model_folder, self.model_name)

    def __init__(self):
        if AnimeIDF._instance is None:
            # 如果模型不存在, 则会下载模型
            if not os.path.exists(self.model_path):
                print(f"Downloading model to {self.model_path}...")
                try:
                    download(AnimeIDF.model_url, folder=self.model_folder.__str__(), filename=self.model_name)
                except Exception as e:
                    print("Model download failed, remote file not found.")
                    raise e
                print(f"Model downloaded in {self.model_path} successfully.")
            try:
                self.session = onnxruntime.InferenceSession(self.model_path)
            except Exception:
                os.remove(self.model_path)
                raise Exception("Model is broken, please try again.")
            AnimeIDF._instance = self
        else:
            self.session = AnimeIDF._instance.session

    @staticmethod
    def _read_image(content: Union[str, IO, TextIOBase]):
        if isinstance(content, str):
            if not os.path.exists(content):
                raise FileNotFoundError(f"File {content} not found.")
            return cv2.imread(content)
        # write to temp file
        with tempfile.NamedTemporaryFile(mode="wb", delete=True) as f:
            f.write(content.read())
            f.seek(0)
            try:
                return cv2.imread(f.name)
            finally:
                f.close()

    def predict_image(self, content: Union[str, IO, TextIOBase]):
        """
        Predict image

        with open("test.png", 'r') as file_io:
            predict_image(file_io)

        :param content: PATH or IO
        :return:
        """
        content.seek(0)
        img = self._read_image(content=content)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225],
                                                                                   dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        ort_inputs = {self.session.get_inputs()[0].name: img}
        ort_outs = self.session.run(None, ort_inputs)

        ort_outs = ort_outs[0][0]
        ort_outs = np.exp(ort_outs) / np.sum(np.exp(ort_outs), axis=0)

        return round(ort_outs[1] * 100, 2)


if __name__ == "__main__":
    model = AnimeIDF()
    print(model.predict_image("anime.jpg"))
