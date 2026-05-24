import numpy as np, onnxruntime as ort
from PIL import Image

class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img, size):
        w, h = img.size
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas.paste(img_r, (pad_x, pad_y))
        return canvas, scale, (pad_x, pad_y)

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        ow, oh = img.size
        x, scale, (px, py) = self._letterbox(img, self.imgsz)
        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out = self.session.run(None, {self.input_name: x})[0][0]  # (300, 6)
        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue
            x1 = max(0.0, min(ow, (x1 - px) / scale))
            y1 = max(0.0, min(oh, (y1 - py) / scale))
            x2 = max(0.0, min(ow, (x2 - px) / scale))
            y2 = max(0.0, min(oh, (y2 - py) / scale))
            results.append({
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results
