import cv2
import yaml
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QDesktopWidget, QTextEdit, \
    QHBoxLayout, QSizePolicy, QFileDialog

from controls.animated_button import AnimatedButton
from pages.login_window import LoginWindow
from utils import logger


class MainWindow(QMainWindow):
    def __init__(self):
        logger.info('初始化主页')

        super().__init__()

        self.login_window = None
        self.video_file = None

        # 读取 YAML 文件
        with open('configs/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # 访问数据
        window_width = data['main']['window_width']
        window_height = data['main']['window_height']

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

        # 创建主布局
        self.main_layout = QVBoxLayout()

        # 创建上传视频按钮
        self.upload_button = QPushButton("上传视频")
        self.upload_button.setFixedSize(100, 30)
        self.upload_button.setStyleSheet("background-color: white; color: black; border-radius: 5px;")
        self.upload_button.clicked.connect(self.upload_video)

        # 创建视频信息展示框
        self.video_info_text = QTextEdit()
        self.video_info_text.setReadOnly(True)
        self.video_info_text.setStyleSheet("background-color: rgba(255, 255, 255, 128); border-radius: 5px;")
        self.main_layout.addWidget(self.video_info_text)  # addWidget()函数将控件添加到当前布局

        # 创建开始运行按钮
        self.run_button = QPushButton("开始运行")
        self.run_button.setFixedHeight(30)  # 设置按钮的高度
        self.run_button.setStyleSheet("background-color: green; color: white; border-radius: 5px;")
        self.run_button.clicked.connect(self.start)
        self.main_layout.addWidget(self.run_button)

        # 创建退出登录按钮
        self.logout_button = QPushButton("退出登录")
        self.logout_button.setFixedSize(100, 30)
        self.logout_button.setStyleSheet("background-color: red; color: white; border-radius: 5px;")
        self.logout_button.clicked.connect(self.logout)

        # 创建主部件并设置布局
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # 创建顶部工具栏
        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置工具栏的大小策略

        # 添加 upload_button 按钮
        self.toolbar.addWidget(self.upload_button)

        # 添加伸缩空间
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # 添加 logout_button 按钮
        self.toolbar.addWidget(self.logout_button)

        # 设置窗口大小策略为固定大小
        self.setFixedSize(self.size())

    def upload_video(self):
        """
        上传视频文件
        :return:
        """
        logger.info('选择要上传的视频文件')

        # 打开文件对话框并获取用户选择的文件路径
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_path:
            logger.info("选择文件路径：" + file_path)

            self.video_file = cv2.VideoCapture(file_path)

            # 获取视频的基本信息
            fps = self.video_file.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_file.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 将视频信息显示到展示框中
            self.video_info_text.append(file_path)
            self.video_info_text.append(f"\n帧率: {fps}")
            self.video_info_text.append(f"总帧数: {frame_count}")
            self.video_info_text.append(f"分辨率: {width}x{height}")

    def start(self):
        """
        开始运行
        :return:
        """
        logger.info('开始运行跟踪程序')
        # TODO: 接入运行逻辑

    def logout(self):
        """
        退出登录
        :return:
        """
        logger.info('当前用户退出登录')

        self.login_window = LoginWindow()
        self.login_window.show()

        self.close()
