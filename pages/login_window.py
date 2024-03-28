import yaml
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from utils import logger
from utils.encryptor import encrypt_password

USERS = {
    'Liu_JM': 'e10adc3949ba59abbe56e057f20f883e'
}


class LoginWindow(QWidget):
    def __init__(self):
        logger.info('初始化登录页面')

        super().__init__()

        self.main_window = None

        # 读取 YAML 文件
        with open('configs/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # 访问数据
        window_width = data['login']['window_width']
        window_height = data['login']['window_height']

        self.setWindowTitle('用户登录')
        self.setFixedSize(window_width, window_height)

        layout = QVBoxLayout()

        # 创建标题标签
        title_label = QLabel('多行人目标跟踪系统')
        title_label.setFont(QFont('Arial', 20))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 创建用户名输入框
        username_layout = QHBoxLayout()
        username_label = QLabel('用户名:')
        self.username_input = QLineEdit()
        self.username_input.setText('Liu_JM')  # 设置默认用户名
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        layout.addLayout(username_layout)

        # 创建密码输入框
        password_layout = QHBoxLayout()
        password_label = QLabel('密码:  ')
        self.password_input = QLineEdit()
        self.password_input.setText('123456')
        self.password_input.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        layout.addLayout(password_layout)

        # 创建按钮
        button_layout = QHBoxLayout()
        login_button = QPushButton('登录')
        register_button = QPushButton('注册')
        forgot_button = QPushButton('忘记密码')
        button_layout.addWidget(login_button)
        button_layout.addWidget(register_button)
        button_layout.addWidget(forgot_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接按钮的点击事件
        login_button.clicked.connect(self.login)
        register_button.clicked.connect(self.register)
        forgot_button.clicked.connect(self.retrieve_password)

        # 连接用户名输入框和密码输入框的回车键事件处理函数
        self.username_input.returnPressed.connect(self.login)
        self.password_input.returnPressed.connect(self.login)

    def login(self):
        """
        执行登录校验
        :return:
        """
        logger.info('登录用户：' + self.username_input.text())

        username = self.username_input.text()
        password = self.password_input.text()
        password = encrypt_password(password)

        # 非空校验
        if not username or not password:
            QMessageBox.warning(self, '警告', '用户名和密码不能为空！', QMessageBox.Ok)
            return

        # 用户名密码校验
        if USERS.get(username) != password:
            logger.info('登录失败')

            QMessageBox.warning(self, '警告', '用户名或密码不匹配，请重试！', QMessageBox.Ok)
            return

        from pages.main_window import MainWindow
        self.main_window = MainWindow()
        self.main_window.show()

        self.close()

    def register(self):
        """
        新用户注册
        :return:
        """
        logger.info('新用户注册')
        # TODO: 执行注册逻辑

    def retrieve_password(self):
        """
        忘记密码重置
        :return:
        """
        logger.info('找回密码，用户：' + self.username_input.text())
        # TODO: 执行找回密码逻辑
