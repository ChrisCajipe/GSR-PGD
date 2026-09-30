from PySide6.QtCore import QObject, Signal


class AttackWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    # NEW: sends the current stage name to the UI
    stageChanged = Signal(str)

    def __init__(self, image, target_class):
        super().__init__()

        self.image = image
        self.target_class = target_class

    def run(self):
        try:
            from backend.attack_runner import generate_attacks

            results = generate_attacks(
                image=self.image,
                target_class=self.target_class,
                progress_callback=self.stageChanged.emit
            )

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))