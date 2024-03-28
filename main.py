import sys

from PyQt5.QtWidgets import QApplication

from pages.login_window import LoginWindow
from pages.main_window import MainWindow
from utils import logger

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = LoginWindow()
    window.show()
    logger.info('启动成功')

    app.exec()
