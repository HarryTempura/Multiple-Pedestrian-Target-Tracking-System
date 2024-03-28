from PyQt5.QtCore import QPropertyAnimation, QRect
from PyQt5.QtWidgets import QPushButton


class AnimatedButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation = QPropertyAnimation(self, b"geometry")

    def mousePressEvent(self, event):
        # 鼠标按下时的缩放动画效果
        start_rect = self.geometry()
        end_rect = start_rect.adjusted(-5, -5, 5, 5)
        self.animate(start_rect, end_rect)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # 鼠标释放时的缩放动画效果
        start_rect = self.geometry()
        end_rect = start_rect.adjusted(5, 5, -5, -5)
        self.animate(start_rect, end_rect)
        super().mouseReleaseEvent(event)

    def animate(self, start_rect, end_rect):
        self.animation.setDuration(100)
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()
