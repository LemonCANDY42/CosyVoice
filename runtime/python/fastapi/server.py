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

from pathlib import Path

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

# 1. 定义自定义异常
class InsufficientFundsError(Exception):
    def __init__(self, message: str, error_code: int):
        self.message = message
        self.error_code = error_code
        super().__init__(f"InsufficientFundsError:{message},error code:{error_code}")

# 2. 为自定义异常注册处理器
@app.exception_handler(InsufficientFundsError)
async def insufficient_funds_handler(request: Request, exc: InsufficientFundsError):
    # 在这里定义遇到此异常时返回给客户端的响应
    return JSONResponse(
        status_code=exc.error_code,  # 你可以使用合适的 HTTP 状态码，或者像本例一样使用非标准码
        content={
            "message": "TTS error",
            "error": exc.message,
            "detail": str(exc)
        },
    )

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))

# 看这里
# 1. 定义 Pydantic 模型来描述请求体结构
class TTSRequest(BaseModel):
    tts_text: str
    zero_shot_spk_id: int
    seed: int
    speed: float = 1.0
    
class MaterialRequest(BaseModel):
    zero_shot_spk_id: int
    type: int
    link: str

@app.get("/v1/tts")
@app.post("/v1/tts")
async def tts(params: TTSRequest):
    """
        请求示意：
        {
            "tts_text": "欢迎使用视追智能的数字人直播系统！",
            "zero_shot_spk_id": 203,
            "seed": 42,
            "speed":1.0
            
        }
    """
    tts_text = params.tts_text
    zero_shot_spk_id = params.zero_shot_spk_id
    seed = params.seed
    speed = params.speed
    # 说话人ID，用int类型
    if not isinstance(zero_shot_spk_id, int):
        raise InsufficientFundsError(message="zero_shot_spk_id is required and must be int", error_code=1004)
    if not isinstance(speed, float):
        raise InsufficientFundsError(message="speed is required and must be float", error_code=1006)
    set_all_random_seed(seed)
    model_output = cosyvoice.inference_zero_shot(tts_text,'', '', zero_shot_spk_id, stream=False, speed=speed)
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
    if materials_type != 3:
        # Only materials_type 3 is supported for tts.
        raise InsufficientFundsError(message=f"unknown type:{materials_type}", error_code=1001)
    if not isinstance(zero_shot_spk_id, int):
        raise InsufficientFundsError(message="zero_shot_spk_id is required and must be int", error_code=1004)
    try:
        prompt_wav = str(Path(link))
        prompt_text = prompt_wav_recognition(prompt_wav)
        prompt_speech_16k = load_wav(prompt_wav, 16000)
    except Exception as e:
        raise InsufficientFundsError(message="load wav/prompt_wav_recognition error:"+str(e), error_code=1002)
    
    if cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id):
        try:
            cosyvoice.save_spkinfo()
        except Exception as e:
            raise InsufficientFundsError(message="save_spkinfo error:"+str(e), error_code=1005)
        return {"message": "Material processed successfully!"}
    else:
        raise InsufficientFundsError(message=f"Failed to process material", error_code=1003)
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

    try:
        cosyvoice = CosyVoice(args.model_dir, spk2info_path=Path(args.spk2info_path))
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir,load_jit=args.load_jit,load_trt=args.load_trt,fp16=args.fp16,spk2info_path=Path(args.spk2info_path))
        except Exception as e:
            raise e

    asr_model = AutoModel(
            model=args.asr_model_path,
            disable_update=True,
            log_level='DEBUG',
            device="cuda:0")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
