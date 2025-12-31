
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

class TextToSpeechModel():
    def __init__(self,
                 model_id = "openai/whisper-large-v3-turbo",
                 device = "cuda:0", # or "cpu"
                 torch_dtype = torch.float16 # torch.float32
                ):
        super(TextToSpeechModel, self).__init__()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )


    def callback(self, msg):
        self.pub.publish(data=self.forward(msg.data))

    def forward(self, file: str = "TestSound"):
        return self.pipe(file)['text']
