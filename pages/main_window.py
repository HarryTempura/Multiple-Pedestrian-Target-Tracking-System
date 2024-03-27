import cv2
import yaml
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QDesktopWidget, QTextEdit, \
    QHBoxLayout, QSizePolicy, QFileDialog

from utils import logger


class MainWindow(QMainWindow):
    def __init__(self):
        logger.info('初始化主页')

        super().__init__()

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
        main_layout = QVBoxLayout()

        # 创建上传视频按钮
        upload_button = QPushButton("上传视频")
        upload_button.setFixedSize(100, 30)
        upload_button.setStyleSheet("background-color: white; color: black; border-radius: 15px;")
        upload_button.clicked.connect(self.upload_video)

        # 创建视频信息展示框
        video_info_text = QTextEdit()
        video_info_text.setReadOnly(True)
        video_info_text.setStyleSheet("background-color: rgba(255, 255, 255, 128); border-radius: 5px;")
        main_layout.addWidget(video_info_text)  # addWidget()函数将控件添加到当前布局

        # 创建开始运行按钮
        run_button = QPushButton("开始运行")
        run_button.setFixedHeight(30)  # 设置按钮的高度
        run_button.setStyleSheet("background-color: green; color: white; border-radius: 5px;")
        run_button.clicked.connect(self.start)
        main_layout.addWidget(run_button)

        # 创建退出登录按钮
        logout_button = QPushButton("退出登录")
        logout_button.setFixedSize(100, 30)
        logout_button.setStyleSheet("background-color: red; color: white; border-radius: 15px;")

        # 创建左上角布局
        top_left_layout = QHBoxLayout()
        top_left_layout.addWidget(logout_button)
        top_left_layout.addStretch()

        # 创建主部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 创建顶部工具栏
        toolbar = self.addToolBar("Toolbar")
        toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置工具栏的大小策略

        # 添加 upload_button 按钮
        toolbar.addWidget(upload_button)

        # 添加伸缩空间
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # 添加 logout_button 按钮
        toolbar.addWidget(logout_button)

        # 设置窗口大小策略为固定大小
        self.setFixedSize(self.size())

    def upload_video(self):
        """
        上传视频文件
        :return:
        """
        # 打开文件对话框并获取用户选择的文件路径
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_path:
            # 用户选择了文件，可以进行进一步处理，例如读取视频文件等
            logger.info("选择文件路径：" + file_path)

            self.video_file = cv2.VideoCapture(file_path)

    def start(self):
        """
        开始运行
        :return:
        """
        pass  # TODO: 接入运行逻辑

    def logout(self):
        """
        退出登录
        :return:
        """
        pass  # TODO: 返回到登录页面并关闭当前页面
