import hashlib


def encrypt_password(password):
    # 创建 MD5 哈希对象
    md5 = hashlib.md5()

    # 更新哈希对象的输入内容为密码的字节串
    md5.update(password.encode('utf-8'))

    # 获取哈希值（经过加密的密码）
    hashed_password = md5.hexdigest()

    return hashed_password
