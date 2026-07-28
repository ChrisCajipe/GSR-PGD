from PySide6.QtCore import QObject, Signal


class AttackWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image, target_class):
        super().__init__()

        self.image = image
        self.target_class = target_class

    def run(self):
        try:
            from backend.attack_runner import generate_attacks

            results = generate_attacks(
                image=self.image,
                target_class=self.target_class
            )

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))