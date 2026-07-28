from torchvision.transforms.functional import to_pil_image
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap

def tensor_to_pixmap(tensor):
    tensor = tensor.detach().cpu()

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    image = to_pil_image(tensor.cpu())

    qt = ImageQt(image)

    return QPixmap.fromImage(ImageQt(image))