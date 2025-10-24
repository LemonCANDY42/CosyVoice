import argparse
from re import T
from server import Server

if __name__ == "__main__":
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
                        default=True)
    parser.add_argument('--asr_model_path',
                        type=str,
                        default='/workspace/CosyVoice/pretrained_models/SenseVoiceSmall',
                        help='放置所有说话人嵌入信息的文件夹路径')
    parser.add_argument('--spk2info_path',
                        type=str,
                        default='/workspace/mnt/data/materials/spk2info/spk2info.pt',#'iic/CosyVoice-300M',
                        help='放置所有说话人嵌入信息的文件夹路径')
    parser.add_argument('--load_vllm',
                        type=bool,
                        default=True,
                        help='是否加载VLLM模型')
    parser.add_argument('--load_trt',
                        type=bool,
                        default=True,
                        help='是否加载TensorRT模型')
    parser.add_argument('--trt_concurrent',
                        type=int,
                        default=1,
                        help='TensorRT并发数')
    parser.add_argument('--fp16',
                        type=bool,
                        default=True,
                        help='是否使用FP16精度')
    args = parser.parse_args()

    tts_server = Server(host="0.0.0.0",
                        port=args.port,
                        model_dir=args.model_dir,
                        load_jit=args.load_jit,
                        load_vllm=args.load_vllm,
                        asr_model_path=args.asr_model_path,
                        spk2info_path=args.spk2info_path,
                        load_trt=args.load_trt,
                        fp16=args.fp16)
    tts_server.run()