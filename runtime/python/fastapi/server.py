# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File,Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import threading
import subprocess
import logging
from pathlib import Path    

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 配置多个NTP服务器（建议至少3个以提高可靠性）
NTP_SERVERS = [
    'pool.ntp.org',
    'time.nist.gov',
    'time.windows.com',
    'ntp.aliyun.com',  # 阿里云NTP服务器
    'cn.pool.ntp.org'  # 中国的NTP服务器池
]

def sync_time_with_ntpdate():
    """
    尝试从配置的NTP服务器列表中同步时间，直到成功或遍历所有服务器。
    使用sudo执行ntpdate命令通常需要管理员权限。
    """
    for server in NTP_SERVERS:
        try:
            # 使用sudo执行ntpdate命令（需要权限）
            # 注意：在某些系统上，可能需要配置免密码sudo或使用其他权限管理方式
            cmd = ['ntpdate', '-u', server]  # -u 参数有助于绕过防火墙
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            logger.info(f"时间同步成功 (服务器: {server}): {result.stdout}")
            return True  # 同步成功则退出
        except subprocess.CalledProcessError as e:
            logger.error(f"服务器 {server} 同步失败，返回码 {e.returncode}: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"与服务器 {server} 的同步请求超时")
        except FileNotFoundError:
            logger.critical("系统中未找到 ntpdate 命令，请先安装 ntpdate 工具。")
            break  # 如果找不到命令，则退出循环
        except Exception as e:
            logger.error(f"尝试与服务器 {server} 同步时发生未知错误: {e}")
    logger.error("所有配置的NTP服务器同步均失败。")
    return False


def periodic_time_sync(interval=3600):
    """
    定期执行时间同步的函数
    :param interval: 同步间隔时间（秒），默认3600秒（1小时）
    """
    while True:
        sync_time_with_ntpdate()
        # 等待下一次同步
        threading.Event().wait(interval)

from trust import TrustValidator

def prompt_wav_recognition(prompt_wav):
    """Recognize text from a prompt wav file.

    Args:
        prompt_wav (str or Path): Path to the prompt wav file.

    Returns:
        text(str): Recognized text from the prompt wav file.
    """
    res = asr_model.generate(input=prompt_wav,
                            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                            use_itn=True,
                            )
    text = res[0]["text"].split('|>')[-1]
    return text
    
app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 在启动时设置共享变量（例如，在应用的构造函数或生命周期事件中）
app.state.shared_token = None  

class LoginRequest(BaseModel):
    token: str
    
SECRET_KEY = "my_super_secret_key_szzn"
SALT = "fixed_salt_value_42"
PRESET_STRING = "trust_verification_1111"

validator = TrustValidator(SECRET_KEY, SALT, PRESET_STRING)

@app.get("/login")
@app.post("/login")
async def login(params: LoginRequest):
    app.state.shared_token = params.token
    if validator.validate_server_token(app.state.shared_token):
        return {"message": "Login successful", "token": app.state.shared_token}
    return {"message": "Invalid credentials"}, 401

# @app.get("/inference_sft")
# @app.post("/inference_sft")
# async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
#     model_output = cosyvoice.inference_sft(tts_text, spk_id)
#     return StreamingResponse(generate_data(model_output))

# 1. 定义自定义异常
class InsufficientFundsError(Exception):
    def __init__(self, detail: str, error_code: int):
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"InsufficientFundsError:{detail},error code:{error_code}")

# 2. 为自定义异常注册处理器
@app.exception_handler(InsufficientFundsError)
async def insufficient_funds_handler(request: Request, exc: InsufficientFundsError):
    # 在这里定义遇到此异常时返回给客户端的响应
    return JSONResponse(
        status_code=exc.error_code,  # 你可以使用合适的 HTTP 状态码，或者像本例一样使用非标准码
        content={
            "detail": exc.detail,
        },
    )
    
def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


# @app.get("/inference_sft")
# @app.post("/inference_sft")
# async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
#     model_output = cosyvoice.inference_sft(tts_text, spk_id)
#     return StreamingResponse(generate_data(model_output))


# @app.get("/inference_zero_shot")
# @app.post("/inference_zero_shot")
# async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
#     prompt_speech_16k = load_wav(prompt_wav.file, 16000)
#     model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
#     return StreamingResponse(generate_data(model_output))


# @app.get("/inference_cross_lingual")
# @app.post("/inference_cross_lingual")
# async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
#     prompt_speech_16k = load_wav(prompt_wav.file, 16000)
#     model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
#     return StreamingResponse(generate_data(model_output))


# @app.get("/inference_instruct")
# @app.post("/inference_instruct")
# async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
#     model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
#     return StreamingResponse(generate_data(model_output))


# @app.get("/inference_instruct2")
# @app.post("/inference_instruct2")
# async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
#     prompt_speech_16k = load_wav(prompt_wav.file, 16000)
#     model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
#     return StreamingResponse(generate_data(model_output))

# 看这里
# 1. 定义 Pydantic 模型来描述请求体结构
class TTSRequest(BaseModel):
    tts_text: str
    zero_shot_spk_id: int
    instruct_text: str | None = None
    seed: int
    speed: float = 1.0
    
class MaterialRequest(BaseModel):
    zero_shot_spk_id: int | list = None
    type: int | None = None
    link: str | None = None

@app.get("/v1/tts")
@app.post("/v1/tts")
async def tts(params: TTSRequest):
    """
        请求示意：
        {
            "tts_text": "欢迎使用视追智能的数字人直播系统！",
            "zero_shot_spk_id": 203,
            "instruct_text": "用带货主播的语气欢快的说",
            "seed": 42,
            "speed":1.0
            
        }
    """
    tts_text = params.tts_text
    zero_shot_spk_id = params.zero_shot_spk_id
    instruct_text = params.instruct_text
    seed = params.seed
    speed = params.speed
    try:
        if validator.validate_server_token(app.state.shared_token):
            # 说话人ID，用int类型
            if not isinstance(zero_shot_spk_id, int):
                raise InsufficientFundsError(detail="zero_shot_spk_id is required and must be int", error_code=400)
            if not isinstance(speed, float):
                raise InsufficientFundsError(detail="speed is required and must be float", error_code=400)
            if zero_shot_spk_id not in cosyvoice.frontend.spk2info.keys():
                raise InsufficientFundsError(detail=f"zero_shot_spk_id {zero_shot_spk_id} not found", error_code=404)
            set_all_random_seed(seed)
            if instruct_text is not None and instruct_text != "":
                print(f"tts_text: {tts_text}, instruct_text: {instruct_text}, zero_shot_spk_id: {zero_shot_spk_id}")
                model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, '', zero_shot_spk_id, stream=False, speed=speed)
            else:
                model_output = cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id, stream=False, speed=speed)
        else:
            raise InsufficientFundsError(detail="Invalid token", error_code=401)
    except Exception as e:
        raise InsufficientFundsError(detail="error:"+str(e), error_code=500)
    return StreamingResponse(generate_data(model_output))

# @app.get("/api/v1/materials")
@app.post("/api/v1/materials")
async def materials(params: MaterialRequest):
    """
        link里的文件必须是wav！！！
        请求示意：
        {
            "zero_shot_spk_id": 203,
            "type": 3,
            "link": "/path/to/audio.wav"
        }
    """
    zero_shot_spk_id = params.zero_shot_spk_id
    materials_type = params.type
    link = params.link
    try:
        if validator.validate_server_token(app.state.shared_token):
            if materials_type != 3:
                # Only materials_type 3 is supported for tts.
                raise InsufficientFundsError(detail=f"unknown type:{materials_type}", error_code=400)
            if not isinstance(zero_shot_spk_id, int):
                raise InsufficientFundsError(detail="zero_shot_spk_id is required and must be int", error_code=400)
            try:
                prompt_wav = str(Path(link))
                prompt_text = prompt_wav_recognition(prompt_wav)
                prompt_speech_16k = load_wav(prompt_wav, 16000)
            except Exception as e:
                raise InsufficientFundsError(detail="load wav/prompt_wav_recognition error:"+str(e), error_code=422)
            
            if cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id):
                try:
                    cosyvoice.save_spkinfo()
                except Exception as e:
                    raise InsufficientFundsError(detail="save_spkinfo error:"+str(e), error_code=422)
                return {"message": "Material processed successfully!"}
            else:
                raise InsufficientFundsError(detail=f"Failed to process material", error_code=422)
        else:
            raise InsufficientFundsError(detail="Invalid token", error_code=401)
    except Exception as e:
        raise InsufficientFundsError(detail="error:"+str(e), error_code=500)

@app.get("/api/v1/materials/list")
async def materials_list(params: MaterialRequest):
    """
        功能：获取tts的说话人列表
        请求示意：
        {
            "type": 3,
        }
    """
    try:
        if validator.validate_server_token(app.state.shared_token):
            materials_type = params.type
            if materials_type != 3:
                # Only materials_type 3 is supported for tts.
                raise InsufficientFundsError(detail=f"unknown type:{materials_type}", error_code=422)
            zero_shot_spk_id_list = cosyvoice.frontend.spk2info.keys()
            return {"zero_shot_spk_id_list": list(zero_shot_spk_id_list)}
        else:
            raise InsufficientFundsError(detail="Invalid token", error_code=401)
    except Exception as e:
        raise InsufficientFundsError(detail="error:"+str(e), error_code=500)

@app.get("/api/v1/materials/remove")
@app.post("/api/v1/materials/remove")
async def materials_remove(params: MaterialRequest):
    """
        功能：删除指定的zero_shot_spk_id,可以是列表
        请求示意：
        {
            "zero_shot_spk_id": 203,
            "type": 3
        }
        或者
        {
            "zero_shot_spk_id": [203,204],
            "type": 3
        }
    """
    materials_type = params.type
    zero_shot_spk_id = params.zero_shot_spk_id
    try:
        if validator.validate_server_token(app.state.shared_token):
            if materials_type != 3:
                # Only materials_type 3 is supported for tts.
                raise InsufficientFundsError(detail=f"unknown type:{materials_type}", error_code=400)
            if not isinstance(zero_shot_spk_id, (int, list)):
                raise InsufficientFundsError(detail="zero_shot_spk_id is required and must be int or list", error_code=400)
            if isinstance(zero_shot_spk_id, int):
                zero_shot_spk_id = [zero_shot_spk_id]
                
            can_remove_spk_id = []
            for spk_id in zero_shot_spk_id:
                if spk_id in cosyvoice.frontend.spk2info.keys():
                    can_remove_spk_id.append(spk_id)    
                else:
                    raise InsufficientFundsError(detail=f"zero_shot_spk_id {spk_id} not found", error_code=404)
            
            for spk_id in can_remove_spk_id:
                cosyvoice.frontend.spk2info.pop(spk_id, None)
            try:
                cosyvoice.save_spkinfo()
            except Exception as e:
                raise InsufficientFundsError(detail="save_spkinfo error:"+str(e), error_code=500)
            return {"message": f"Material {can_remove_spk_id} removed successfully!"}
        else:
            raise InsufficientFundsError(detail="Invalid token", error_code=401)
    except Exception as e:
        raise InsufficientFundsError(detail="error:"+str(e), error_code=500)

class Server:
    def __init__(self, **kwargs):
        self.host = kwargs.get("host", "0.0.0.0")
        self.port = kwargs.get("port", 8000)
        self.asr_model_path = kwargs.get("asr_model_path", "/workspace/CosyVoice/pretrained_models/SenseVoiceSmall")
        self.model_dir = kwargs.get("model_dir", "/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
        self.load_jit = kwargs.get("load_jit", False)
        self.load_trt = kwargs.get("load_trt", True)
        self.fp16 = kwargs.get("fp16", True)
        self.spk2info_path = kwargs.get("spk2info_path", "/workspace/mnt/data/materials/spk2info/spk2info.pt")
    def run(self):
        global cosyvoice, asr_model
        # 创建并启动时间同步线程
        # 设置为守护线程，这样当主程序退出时，线程也会退出
        sync_thread = threading.Thread(target=periodic_time_sync, daemon=True)
        sync_thread.start()
        logger.info("时间同步线程已启动。")

        # 启动 FastAPI 应用
        # 使用 uvicorn 运行应用，指定主机和端口
        logger.info("启动 FastAPI 应用...")

        try:
            cosyvoice = CosyVoice(self.model_dir, spk2info_path=Path(self.spk2info_path))
        except Exception:
            try:
                cosyvoice = CosyVoice2(self.model_dir,load_jit=self.load_jit,load_trt=self.load_trt,fp16=self.fp16,spk2info_path=Path(self.spk2info_path))
            except Exception as e:
                raise e

        asr_model = AutoModel(
                model=self.asr_model_path,
                disable_update=True,
                log_level='DEBUG',
                device="cuda:0")

        uvicorn.run(app, host=self.host, port=self.port)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B',#'iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    # load jit 和 load trt 互斥
    parser.add_argument('--load_jit',
                        type=bool,
                        default=False)
    parser.add_argument('--asr_model_path',
                        type=str,
                        default='/workspace/CosyVoice/pretrained_models/SenseVoiceSmall',
                        help='放置所有说话人嵌入信息的文件夹路径')
    parser.add_argument('--spk2info_path',
                        type=str,
                        default='/workspace/mnt/data/materials/spk2info/spk2info.pt',#'iic/CosyVoice-300M',
                        help='放置所有说话人嵌入信息的文件夹路径')
    parser.add_argument('--load_trt',
                        type=bool,
                        default=True,
                        help='是否加载TensorRT模型')
    parser.add_argument('--fp16',
                        type=bool,
                        default=True,
                        help='是否使用FP16精度')
    args = parser.parse_args()

    tts_server = Server(host="0.0.0.0",
                        port=args.port,
                        model_dir=args.model_dir,
                        load_jit=args.load_jit,
                        asr_model_path=args.asr_model_path,
                        spk2info_path=args.spk2info_path,
                        load_trt=args.load_trt,
                        fp16=args.fp16)
    tts_server.run()
