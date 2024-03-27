import logging
import sys

# 创建日志记录器
logger = logging.getLogger(__name__)
# 设置日志记录级别为 INFO
logger.setLevel(logging.INFO)

# 创建一个文件处理器，将日志记录到文件中或者直接在终端输出
# file_handler = logging.FileHandler('system.log')
file_handler = logging.StreamHandler(sys.stdout)
# 创建一个格式化器，定义日志记录的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 将格式化器设置到处理器中
file_handler.setFormatter(formatter)
# 将处理器添加到日志记录器中
logger.addHandler(file_handler)
