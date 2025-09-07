"""
FastAPI信任验证模块
实现客户端加密和服务端解密的登录验证机制
"""
import base64
import time
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import hashlib

class TrustValidator:
    """信任验证器，用于客户端加密和服务端解密验证"""
    
    def __init__(self, secret_key: str, salt: str, preset_string: str):
        """
        初始化信任验证器
        
        Args:
            secret_key: 加密密钥
            salt: 加盐字符串
            preset_string: 预设的验证字符串
        """
        self.secret_key = secret_key
        self.salt = salt
        self.preset_string = preset_string
        self.time_salt = 100860  # 时间盐，防止时间戳被猜测
        
        # 生成AES密钥（使用SHA256哈希确保32字节长度）
        key_material = f"{secret_key}{salt}".encode('utf-8')
        self.aes_key = hashlib.sha256(key_material).digest()
    
    def encrypt_token(self, timestamp: float = None) -> str:
        """
        加密令牌 - 客户端使用
        
        Args:
            timestamp: 时间戳，如果为None则使用当前时间
            
        Returns:
            str: Base64编码的加密令牌
        """
        if timestamp is None:
            timestamp = time.time()+self.time_salt
        
        # 构造明文：时间戳 + 预设字符串
        plaintext = f"{timestamp}:{self.preset_string}"
        plaintext_bytes = plaintext.encode('utf-8')
        
        # 使用AES CBC模式加密
        iv = get_random_bytes(16)
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext_bytes, AES.block_size))
        
        # 组合IV和密文并进行Base64编码
        encrypted_data = iv + ciphertext
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_and_validate(self, encrypted_token: str, timeout: int = 60*60*4) -> bool:
        """
        解密并验证令牌 - 服务端使用
        
        Args:
            encrypted_token: Base64编码的加密令牌
            timeout: 令牌超时时间（秒）
            
        Returns:
            bool: 验证是否成功
        """
        try:
            # Base64解码
            encrypted_data = base64.b64decode(encrypted_token)
            
            # 提取IV和密文
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            
            # 使用AES CBC模式解密
            cipher = AES.new(self.aes_key, AES.MODE_CBC, iv)
            decrypted_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
            decrypted_text = decrypted_bytes.decode('utf-8')
            
            # 解析时间戳和验证字符串
            parts = decrypted_text.split(':', 1)
            if len(parts) != 2:
                return False
                
            timestamp_str, validation_string = parts
            
            # 验证时间戳是否在有效期内
            try:
                timestamp = float(timestamp_str)
                current_time = time.time()+self.time_salt
                if abs(current_time - timestamp) > timeout:
                    return False
            except ValueError:
                return False
            
            # 验证字符串是否匹配预设值
            return validation_string == self.preset_string
            
        except (ValueError, Exception):
            # 处理各种解密和验证错误
            if token is None or token == "":
                return False
            return False
    
    def generate_client_token(self) -> str:
        """生成客户端令牌（简化接口）"""
        return self.encrypt_token()
    
    def validate_server_token(self, token: str) -> bool:
        """验证服务端令牌（简化接口）"""
        return self.decrypt_and_validate(token)


# 示例用法
if __name__ == "__main__":
    # 配置参数（在实际应用中应从配置文件中读取）
    SECRET_KEY = "my_super_secret_key_szzn"
    SALT = "fixed_salt_value_42"
    PRESET_STRING = "trust_verification_1111"
    
    validator = TrustValidator(SECRET_KEY, SALT, PRESET_STRING)
    
    # 客户端生成令牌
    token = validator.generate_client_token()
    # print(f"Generated token: {token}")
    
    # 服务端验证令牌
    is_valid = validator.validate_server_token(token)
    # print(f"Token validation result: {is_valid}")
    
    # 测试无效令牌
    invalid_token = "invalid_base64_string"
    is_valid_invalid = validator.validate_server_token(invalid_token)
    # print(f"Invalid token validation result: {is_valid_invalid}")
