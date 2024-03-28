import random

import yaml
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, \
    QDesktopWidget
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush
from PyQt5.QtCore import Qt

from pages.login import LoginWindow
from utils import logger


class ResetPasswordWindow(QWidget):
    def __init__(self, position=None):
        logger.info('初始化找回页面')

        super().__init__()

        self.phone_num = None
        self.captcha = None
        self.login_window = None

        # 读取 YAML 文件
        with open('configs/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # 访问数据
        window_width = data['generic']['window_width']
        window_height = data['generic']['window_height']
        background = data['background']['image']

        self.setWindowTitle('找回密码')
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

        # 创建手机号输入框
        phone_layout = QHBoxLayout()
        phone_label = QLabel('手  机  号:')
        phone_label.setStyleSheet('color: white')
        self.phone_input = QLineEdit()
        phone_layout.addWidget(phone_label)
        phone_layout.addWidget(self.phone_input)
        layout.addLayout(phone_layout)

        # 创建验证码输入框和获取验证码按钮
        captcha_layout = QHBoxLayout()
        captcha_label = QLabel('验  证  码:')
        captcha_label.setStyleSheet('color: white')
        self.captcha_input = QLineEdit()
        get_captcha_button = QPushButton('获取验证码')
        captcha_layout.addWidget(captcha_label)
        captcha_layout.addWidget(self.captcha_input)
        captcha_layout.addWidget(get_captcha_button)
        layout.addLayout(captcha_layout)

        # 创建新密码输入框
        new_password_layout = QHBoxLayout()
        new_password_label = QLabel('新  密  码:')
        new_password_label.setStyleSheet('color: white')
        self.new_password_input = QLineEdit()
        self.new_password_input.setEchoMode(QLineEdit.Password)
        new_password_layout.addWidget(new_password_label)
        new_password_layout.addWidget(self.new_password_input)
        layout.addLayout(new_password_layout)

        # 创建确认密码输入框
        confirm_password_layout = QHBoxLayout()
        confirm_password_label = QLabel('确认密码:')
        confirm_password_label.setStyleSheet('color: white')
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        confirm_password_layout.addWidget(confirm_password_label)
        confirm_password_layout.addWidget(self.confirm_password_input)
        layout.addLayout(confirm_password_layout)

        # 创建按钮
        button_layout = QHBoxLayout()
        reset_button = QPushButton('重置密码')
        cancel_button = QPushButton('取消')
        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接按钮的点击事件
        get_captcha_button.clicked.connect(self.get_captcha)
        reset_button.clicked.connect(self.reset_password)
        cancel_button.clicked.connect(self.cancel)

    def get_captcha(self):
        """
        获取验证码
        :return:
        """
        self.phone_num = self.phone_input.text()

        # 非空校验
        if not self.phone_num:
            QMessageBox.warning(self, '提示', '电话号不能为空！', QMessageBox.Ok)
            return

        logger.info(f'获取验证码，手机号：{self.phone_input.text()}')

        # 生成四位随机整数
        self.captcha = str(random.randint(1000, 9999))
        logger.info(f'验证码：{self.captcha}')

    def reset_password(self):
        """
        重置密码
        :return:
        """
        logger.info('重置密码')

        phone = self.phone_input.text()
        captcha = self.captcha_input.text()
        new_password = self.new_password_input.text()
        confirm_password = self.confirm_password_input.text()

        # 非空校验
        if phone is None and captcha is None and new_password is None and confirm_password is None:
            QMessageBox.warning(self, '重置失败', '手机号、验证码和密码不能为空')
            return

        # 验证码一致性校验
        if phone != self.phone_num or captcha != self.captcha:
            QMessageBox.warning(self, '重置失败', '验证码不正确')
            return

        if new_password != confirm_password:
            QMessageBox.warning(self, '重置失败', '两次输入的密码不一致')
            return

        # TODO: 执行重置密码逻辑

    def cancel(self):
        """
        取消密码重置
        :return:
        """
        logger.info('取消重置密码')

        self.login_window = LoginWindow(self.pos())
        self.login_window.show()

        self.close()
