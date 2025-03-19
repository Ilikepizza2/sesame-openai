import os
import time
import torch
import torchaudio
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import hf_hub_download
from csm_mlx import CSM, csm_1b, generate, Segment
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model
print("Initializing CSM model...")
start_time = time.time()
csm = CSM(csm_1b())
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# API Models
class ChatMessage(BaseModel):
    role: str
    text: str

class AudioOptions(BaseModel):
    speakers: List[int]  # Speaker IDs for each message
    format: str = "wav"
    temp: float = 0.2
    min_p: float = 0.7
    max_audio_lens: List[int]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    audio: AudioOptions  # New: Contains speakers list & generation options

class AudioData(BaseModel):
    data: str  # Base64 encoded audio

class ResponseMessage(BaseModel):
    role: str = "assistant"
    text: str
    audio: Optional[AudioData] = None

class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    combined_audio: str  # Base64 full conversation

# API Endpoint for conversation generation
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        if len(request.messages) != len(request.audio.speakers):
            raise HTTPException(status_code=400, detail="Number of messages must match number of speakers.")
        
        if len(request.messages) != len(request.audio.max_audio_lens):
            raise HTTPException(status_code=400, detail="Number of messages must match number of max_audio_len.")

        output_dir = "outputs/conversation"
        os.makedirs(output_dir, exist_ok=True)

        context = []
        responses = []
        
        for i, (msg, speaker, max_audio_len) in enumerate(zip(request.messages, request.audio.speakers, request.audio.max_audio_lens)):
            print(f"\nGenerating Turn {i+1}: Speaker {speaker} says '{msg.text}'")
            gen_start = time.time()

            # Generate audio
            audio = generate(
                csm,
                text=msg.text,
                speaker=speaker,
                context=context,
                max_audio_length_ms=max_audio_len,
                sampler=make_sampler(temp=request.audio.temp, min_p=request.audio.min_p)
            )

            gen_time = time.time() - gen_start
            audio_array = np.asarray(audio)

            # Save audio
            output_path = f"{output_dir}/turn_{i+1}_speaker_{speaker}.wav"
            torchaudio.save(output_path, torch.Tensor(audio_array).unsqueeze(0).cpu(), 24000, format="wav")

            # Convert to Base64
            with open(output_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            responses.append(Choice(
                message=ResponseMessage(
                    text=msg.text,
                    audio=AudioData(data=audio_b64)
                )
            ))

            # Store context
            context.append(Segment(
                speaker=speaker,
                text=msg.text,
                audio=mx.array(audio_array)
            ))

        # Concatenate conversation
        combined_audio_b64 = concatenate_conversation(output_dir, request.messages, request.audio.speakers)

        return JSONResponse(content=ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=responses,
            combined_audio=combined_audio_b64
        ).dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def concatenate_conversation(output_dir, messages, speakers):
    """Concatenates all turns into a single audio file."""
    print("\nConcatenating full conversation...")
    all_audio = []
    sample_rate = 24000
    pause_samples = int(0.5 * sample_rate)  # 0.5-second pause
    pause = torch.zeros(1, pause_samples)

    for i, (msg, speaker) in enumerate(zip(messages, speakers)):
        audio_path = f"{output_dir}/turn_{i+1}_speaker_{speaker}.wav"
        waveform, _ = torchaudio.load(audio_path)
        all_audio.append(waveform)

        # Add pause except after last turn
        if i < len(messages) - 1:
            all_audio.append(pause)

    # Concatenate all turns
    full_conversation = torch.cat(all_audio, dim=1)
    output_path = f"{output_dir}/full_conversation.wav"
    torchaudio.save(output_path, full_conversation, sample_rate)

    # Convert to Base64
    with open(output_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)