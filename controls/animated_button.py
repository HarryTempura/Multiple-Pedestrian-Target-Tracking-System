from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import Qt, QPoint, QPropertyAnimation


class AnimatedButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("shadow", 0)
        self.animation = QPropertyAnimation(self, b"shadow")
        self.shadow = 0

    def mousePressEvent(self, event):
        self.shadow = 30
        self.animateShadow(0, 30)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.shadow = 0
        self.animateShadow(30, 0)
        super().mouseReleaseEvent(event)

    def animateShadow(self, start_value, end_value):
        self.animation.setDuration(100)
        self.animation.setStartValue(start_value)
        self.animation.setEndValue(end_value)
        self.animation.start()

    def setShadow(self, value):
        self.shadow = value
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        color = QColor(0, 0, 0, self.shadow)
        qp.setBrush(color)
        qp.drawRoundedRect(self.rect(), 5, 5)
