from torchvision.transforms.functional import to_pil_image
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap


def tensor_to_pil(tensor):
    """
    Converts tensor [C,H,W] into PIL Image.
    """

    tensor = tensor.detach().cpu()

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    return to_pil_image(tensor)


def tensor_to_pixmap(tensor):
    image = tensor_to_pil(tensor)

    return QPixmap.fromImage(
        ImageQt(image)
    )