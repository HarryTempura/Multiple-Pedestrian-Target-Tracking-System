import sys

from PyQt5.QtWidgets import QApplication

from pages.main_window import MainWindow
from utils import logger

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    logger.info('启动成功')

    app.exec()
