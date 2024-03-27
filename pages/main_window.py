import yaml
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QDesktopWidget

from utils import logger


class MainWindow(QMainWindow):
    def __init__(self):
        logger.info('初始化主页')

        super().__init__()

        # 读取 YAML 文件
        with open('configs/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # 访问数据
        window_width = data['general']['window_width']
        window_height = data['general']['window_height']

        self.setWindowTitle('多行人目标跟踪系统')

        # 设置窗口大小
        self.resize(window_width, window_height)

        # 计算窗口在屏幕上的位置
        screen_geometry = QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 3
        self.move(x, y)

        # 设置背景图片
        background_image = QPixmap("data/images/MOSS.jpeg")
        palette = self.palette()
        palette.setBrush(QPalette.Background, QBrush(background_image.scaled(self.size())))
        self.setPalette(palette)

        # 创建一个垂直布局对象
        layout = QVBoxLayout()

        label = QLabel('Hello, PyQt5!')
        label.setFont(QFont('Arial', 24))
        # layout.addWidget 将标控件加到布局中
        layout.addWidget(label)

        # 创建一个按钮控件，并设置文本为 'Click Me'
        button = QPushButton('Click Me')
        layout.addWidget(button)

        # 创建一个 QWidget 控件作为布局的容器
        container = QWidget()
        # 将布局设置为容器的布局
        container.setLayout(layout)
        # 将容器设置为主窗口的中央部件
        self.setCentralWidget(container)

        # 设置窗口大小策略为固定大小
        # self.setFixedSize(self.size())
