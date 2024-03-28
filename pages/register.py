import yaml
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, \
    QDesktopWidget
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush
from PyQt5.QtCore import Qt

from pages.login import LoginWindow
from utils import logger


class RegisterWindow(QWidget):
    def __init__(self, position=None):
        logger.info('初始化注册页面')

        super().__init__()

        self.login_window = None

        # 读取 YAML 文件
        with open('configs/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # 访问数据
        window_width = data['generic']['window_width']
        window_height = data['generic']['window_height']
        background = data['background']['image']

        self.setWindowTitle('用户注册')
        self.setFixedSize(window_width, window_height)

        if position is None:
            # 计算窗口在屏幕上的位置
            screen_geometry = QDesktopWidget().screenGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 3
            self.move(x, y)
        else:
            self.move(position)

        # 设置背景图片
        background_image = QPixmap(background)
        palette = self.palette()
        palette.setBrush(QPalette.Background, QBrush(background_image.scaled(self.size())))
        self.setPalette(palette)

        layout = QVBoxLayout()

        # 创建标题标签
        title_label = QLabel('多行人目标跟踪系统')
        title_label.setFont(QFont('Arial', 20))
        title_label.setStyleSheet('color: white')
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 创建用户名输入框
        username_layout = QHBoxLayout()
        username_label = QLabel('用  户  名:')
        username_label.setStyleSheet('color: white')
        self.username_input = QLineEdit()
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        layout.addLayout(username_layout)

        # 创建密码输入框
        password_layout = QHBoxLayout()
        password_label = QLabel('密       码:  ')
        password_label.setStyleSheet('color: white')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        layout.addLayout(password_layout)

        # 创建确认密码输入框
        confirm_layout = QHBoxLayout()
        confirm_label = QLabel('确认密码:')
        confirm_label.setStyleSheet('color: white')
        self.confirm_input = QLineEdit()
        self.confirm_input.setEchoMode(QLineEdit.Password)
        confirm_layout.addWidget(confirm_label)
        confirm_layout.addWidget(self.confirm_input)
        layout.addLayout(confirm_layout)

        # 创建按钮
        button_layout = QHBoxLayout()
        register_button = QPushButton('注册')
        cancel_button = QPushButton('取消')
        button_layout.addWidget(register_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接按钮的点击事件
        register_button.clicked.connect(self.register)
        cancel_button.clicked.connect(self.cancel)

    def register(self):
        """
        用户信息注册并保存
        :return:
        """
        username = self.username_input.text()
        password = self.password_input.text()
        confirm_password = self.confirm_input.text()

        # 非空校验
        if not username or not password or not confirm_password:
            QMessageBox.warning(self, '提示', '用户名、密码不能为空！', QMessageBox.Ok)
            return

        logger.info('用户注册，用户名：' + self.username_input.text())

        if password != confirm_password:
            QMessageBox.warning(self, '注册失败', '两次输入的密码不一致')
        else:
            self.login_window = LoginWindow(self.pos())
            self.login_window.show()

            self.close()

    def cancel(self):
        """
        取消注册
        :return:
        """
        logger.info('注册取消')

        self.login_window = LoginWindow(self.pos())
        self.login_window.show()

        self.close()
