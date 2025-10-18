#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 服务压力测试工具
用于验证 TensorRT 资源泄漏修复效果

使用方法：
    python stress_test.py --host localhost --port 8000 --total 1000 --concurrent 5
"""

import argparse
import requests
import time
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import random
import statistics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s'
)
logger = logging.getLogger(__name__)

from trust import TrustValidator
SECRET_KEY = "my_super_secret_key_szzn"
SALT = "fixed_salt_value_42"
PRESET_STRING = "trust_verification_1111"

validator = TrustValidator(SECRET_KEY, SALT, PRESET_STRING)

# 客户端生成令牌
TOKEN = validator.generate_client_token()


class TTSStressTester:
    """TTS 服务压力测试器"""
    
    def __init__(self, host='localhost', port=8000, token=None):
        self.base_url = f"http://{host}:{port}"
        self.token = token
        
        # 统计数据
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'timeout': 0,
            'error_types': defaultdict(int),
            'response_times': [],
            'start_time': None,
            'end_time': None
        }
        
        self.lock = threading.Lock()
        
        # 测试文本样本
        self.normal_texts = [
            "欢迎使用视追智能的数字人直播系统！",
            "今天天气真不错，适合出去走走。",
            "人工智能正在改变我们的生活方式。",
            "这是一个测试语音合成的句子。",
            "科技让生活更美好，创新驱动未来发展。",
            "大家好，我是你的AI助手，很高兴为您服务。",
            "语音合成技术在不断进步和发展。",
            "请问有什么我可以帮助您的吗？",
            "感谢您使用我们的服务，祝您使用愉快！",
            "TensorRT加速让推理速度更快更稳定。"
        ]
        
        self.instruct_texts = [
            "用带货主播快速激情的语气",
            "用温柔甜美的声音",
            "用严肃正式的语气",
            "用活泼开朗的tone",
            "用专业播音员的腔调"
        ]
    
    def login(self):
        """登录获取token（如果需要）"""
        if not self.token:
            logger.info("未提供token，跳过登录")
            return True
        
        try:
            url = f"{self.base_url}/login"
            response = requests.post(url, json={"token": self.token}, timeout=10)
            if response.status_code == 200:
                logger.info("登录成功")
                return True
            else:
                logger.error(f"登录失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"登录异常: {e}")
            return False
    
    def create_normal_request(self):
        """创建正常的请求"""
        return {
            "tts_text": random.choice(self.normal_texts),
            "zero_shot_spk_id": 0,
            "seed": random.randint(1, 10000),
            "speed": random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
        }
    
    def create_normal_request_with_instruct(self):
        """创建带指令的正常请求"""
        req = self.create_normal_request()
        req["instruct_text"] = random.choice(self.instruct_texts)
        return req
    
    def create_abnormal_request(self):
        """创建异常请求（用于测试错误处理）"""
        abnormal_types = [
            # 1. 无效的 speaker ID
            {
                "tts_text": "测试文本",
                "zero_shot_spk_id": 99999,  # 不存在的ID
                "seed": 42,
                "speed": 1.0
            },
            # 2. 错误的数据类型
            {
                "tts_text": "测试文本",
                "zero_shot_spk_id": "invalid",  # 应该是int
                "seed": 42,
                "speed": 1.0
            },
            # 3. 超长文本（可能导致推理问题）
            {
                "tts_text": "测试" * 100,  # 超长文本
                "zero_shot_spk_id": 0,
                "seed": 42,
                "speed": 1.0
            },
            # 4. 空文本
            {
                "tts_text": "",
                "zero_shot_spk_id": 0,
                "seed": 42,
                "speed": 1.0
            },
            # 5. 错误的 speed 类型
            {
                "tts_text": "测试文本",
                "zero_shot_spk_id": 0,
                "seed": 42,
                "speed": "fast"  # 应该是 float
            },
            # 6. 无效tts文本（无法被正确解析为token）
            {
                "tts_text": "*&^%$#@*()_+-=[]{}|\\:;\"'<>,.?/~`Ελληνικό αλφάβητο",
                "zero_shot_spk_id": 0,
                "seed": 42,
                "speed": 1.0
            }
        ]
        return random.choice(abnormal_types)
    
    def send_request(self, request_data, request_id, expect_success=True):
        """
        发送单个请求
        
        Args:
            request_data: 请求数据
            request_id: 请求ID
            expect_success: 是否期望成功
            
        Returns:
            dict: 包含结果信息的字典
        """
        url = f"{self.base_url}/v1/tts"
        start_time = time.time()
        
        result = {
            'id': request_id,
            'success': False,
            'response_time': 0,
            'status_code': None,
            'error': None,
            'expect_success': expect_success
        }
        
        try:
            response = requests.post(
                url,
                json=request_data,
                timeout=60,  # 60秒超时
                stream=True
            )
            
            result['status_code'] = response.status_code
            result['response_time'] = time.time() - start_time
            
            if response.status_code == 200:
                # 读取流式响应
                audio_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    audio_size += len(chunk)
                
                result['success'] = True
                result['audio_size'] = audio_size
                
                if expect_success:
                    logger.info(f"✓ Request #{request_id} 成功 - "
                              f"用时: {result['response_time']:.2f}s, "
                              f"音频: {audio_size} bytes")
                else:
                    logger.warning(f"⚠ Request #{request_id} 异常请求却成功了")
            else:
                result['error'] = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    result['error_detail'] = error_detail
                except:
                    pass
                
                if expect_success:
                    logger.error(f"✗ Request #{request_id} 失败 - "
                               f"状态码: {response.status_code}, "
                               f"用时: {result['response_time']:.2f}s")
                else:
                    result['success'] = True
                    logger.info(f"✓ Request #{request_id} 异常请求正确返回错误")
                    
        except requests.Timeout:
            result['response_time'] = time.time() - start_time
            result['error'] = 'Timeout'
            logger.error(f"✗ Request #{request_id} 超时！用时: {result['response_time']:.2f}s")
            
        except Exception as e:
            result['response_time'] = time.time() - start_time
            result['error'] = str(e)
            logger.error(f"✗ Request #{request_id} 异常: {e}, "
                       f"用时: {result['response_time']:.2f}s")
        
        return result
    
    def update_stats(self, result):
        """更新统计信息"""
        with self.lock:
            self.stats['total'] += 1
            
            if result['success']:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
                
                if result['error'] == 'Timeout':
                    self.stats['timeout'] += 1
                
                error_type = result.get('error', 'Unknown')
                self.stats['error_types'][error_type] += 1
            
            if result['response_time'] > 0:
                self.stats['response_times'].append(result['response_time'])
    
    def run_sequential_test(self, total_requests, normal_ratio=0.8):
        """
        顺序测试
        
        Args:
            total_requests: 总请求数
            normal_ratio: 正常请求比例 (0-1)
        """
        logger.info(f"开始顺序测试: 共 {total_requests} 个请求")
        logger.info(f"正常请求比例: {normal_ratio*100}%")
        
        self.stats['start_time'] = datetime.now()
        
        for i in range(total_requests):
            # 根据比例决定是否创建异常请求
            if random.random() < normal_ratio:
                # 50% 普通请求，50% 带指令的请求
                if random.random() < 0.5:
                    request_data = self.create_normal_request()
                else:
                    request_data = self.create_normal_request_with_instruct()
                expect_success = True
            else:
                request_data = self.create_abnormal_request()
                expect_success = False
            
            result = self.send_request(request_data, i + 1, expect_success)
            self.update_stats(result)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                self.print_progress()
        
        self.stats['end_time'] = datetime.now()
        self.print_final_report()
    
    def run_concurrent_test(self, total_requests, concurrent=5, normal_ratio=0.8):
        """
        并发测试
        
        Args:
            total_requests: 总请求数
            concurrent: 并发数
            normal_ratio: 正常请求比例
        """
        logger.info(f"开始并发测试: 共 {total_requests} 个请求, 并发数: {concurrent}")
        logger.info(f"正常请求比例: {normal_ratio*100}%")
        
        self.stats['start_time'] = datetime.now()
        
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = []
            
            for i in range(total_requests):
                if random.random() < normal_ratio:
                    if random.random() < 0.5:
                        request_data = self.create_normal_request()
                    else:
                        request_data = self.create_normal_request_with_instruct()
                    expect_success = True
                else:
                    request_data = self.create_abnormal_request()
                    expect_success = False
                
                future = executor.submit(
                    self.send_request,
                    request_data,
                    i + 1,
                    expect_success
                )
                futures.append(future)
            
            # 等待所有请求完成
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                self.update_stats(result)
                completed += 1
                
                if completed % 10 == 0:
                    self.print_progress()
        
        self.stats['end_time'] = datetime.now()
        self.print_final_report()
    
    def print_progress(self):
        """打印进度信息"""
        with self.lock:
            total = self.stats['total']
            success = self.stats['success']
            failed = self.stats['failed']
            timeout = self.stats['timeout']
            
            success_rate = (success / total * 100) if total > 0 else 0
            
            logger.info(f"📊 进度: {total} | "
                       f"成功: {success} ({success_rate:.1f}%) | "
                       f"失败: {failed} | "
                       f"超时: {timeout}")
    
    def print_final_report(self):
        """打印最终报告"""
        stats = self.stats
        
        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        qps = stats['total'] / duration if duration > 0 else 0
        
        print("\n" + "="*80)
        print("🎯 压力测试最终报告")
        print("="*80)
        print(f"⏰ 测试时间: {stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{stats['end_time'].strftime('%H:%M:%S')}")
        print(f"⏱️  总耗时: {duration:.2f} 秒")
        print(f"📝 总请求数: {stats['total']}")
        print(f"✅ 成功: {stats['success']} ({success_rate:.2f}%)")
        print(f"❌ 失败: {stats['failed']}")
        print(f"⏳ 超时: {stats['timeout']}")
        print(f"🚀 QPS: {qps:.2f} 请求/秒")
        
        if stats['response_times']:
            print(f"\n📈 响应时间统计:")
            print(f"   最小值: {min(stats['response_times']):.3f}s")
            print(f"   最大值: {max(stats['response_times']):.3f}s")
            print(f"   平均值: {statistics.mean(stats['response_times']):.3f}s")
            print(f"   中位数: {statistics.median(stats['response_times']):.3f}s")
            if len(stats['response_times']) > 1:
                print(f"   标准差: {statistics.stdev(stats['response_times']):.3f}s")
        
        if stats['error_types']:
            print(f"\n❌ 错误类型统计:")
            for error_type, count in sorted(stats['error_types'].items(), 
                                           key=lambda x: x[1], 
                                           reverse=True):
                print(f"   {error_type}: {count} 次")
        
        # 判断测试结果
        print("\n" + "="*80)
        if stats['timeout'] > 0:
            print("⚠️  警告: 检测到超时请求！可能存在资源泄漏问题！")
        elif success_rate > 95 and stats['total'] >= 100:
            print("✅ 测试通过: 服务运行稳定，未发现明显问题")
        elif success_rate > 80:
            print("⚠️  警告: 成功率偏低，建议检查错误日志")
        else:
            print("❌ 测试失败: 服务存在严重问题")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='TTS 服务压力测试工具')
    parser.add_argument('--host', type=str, default='localhost',
                       help='服务器地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                       help='服务器端口 (默认: 8000)')
    parser.add_argument('--token', type=str, default=TOKEN,
                       help='认证 token')
    parser.add_argument('--total', type=int, default=100,
                       help='总请求数 (默认: 100)')
    parser.add_argument('--concurrent', type=int, default=1,
                       help='并发数，1为顺序执行 (默认: 1)')
    parser.add_argument('--normal-ratio', type=float, default=0.6,
                       help='正常请求比例 0-1 (默认: 0.8)')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['sequential', 'concurrent', 'auto'],
                       help='测试模式: sequential(顺序), concurrent(并发), auto(自动) (默认: auto)')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = TTSStressTester(host=args.host, port=args.port, token=args.token)
    
    # 登录（如果需要）
    if args.token:
        if not tester.login():
            logger.error("登录失败，退出测试")
            return
    
    print("\n" + "="*80)
    print("🚀 TTS 服务压力测试开始")
    print("="*80)
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"总请求数: {args.total}")
    print(f"并发数: {args.concurrent}")
    print(f"正常请求比例: {args.normal_ratio * 100}%")
    print(f"测试模式: {args.mode}")
    print("="*80 + "\n")
    
    # 选择测试模式
    if args.mode == 'sequential' or (args.mode == 'auto' and args.concurrent == 1):
        tester.run_sequential_test(args.total, args.normal_ratio)
    else:
        tester.run_concurrent_test(args.total, args.concurrent, args.normal_ratio)


if __name__ == '__main__':
    main()


