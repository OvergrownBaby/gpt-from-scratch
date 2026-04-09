import gradio as gr
from model import LLM
import threading
import time
import torch

# import os
# import urllib.request
# if not os.path.exists("shakespeare.txt"):
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     urllib.request.urlretrieve(url, "shakespeare.txt")

# with open("shakespeare.txt", "r") as f:
#     corpus = f.read()

from datasets import load_dataset
import tiktoken
from random import randint

enc = tiktoken.get_encoding("gpt2")
openwebtext_ds = iter(load_dataset("openwebtext", split="train", streaming=True))

gpt = LLM(batch_size=32, sample_len=256, d_model=256, d_k=64, n_layers=6, lr=3e-4)

losses = []
training = True

stats = {"iter": 0, "loss": 0, "best_loss": float("inf"), "iter_per_sec": 0, "tokens_per_sec": 0, "total_tokens": 0, "elapsed": 0, "lr": 1e-3}

def get_batch_from_stream(ds):
    batch = []
    while len(batch) < gpt.batch_size:
        tokens = enc.encode(next(ds)["text"])
        if len(tokens) >= gpt.sample_len + 1:
            start = randint(0, len(tokens) - gpt.sample_len - 1)
            batch.append(tokens[start:start + gpt.sample_len + 1])
    return batch

def train_loop():
    global training
    i = 0
    start = time.time()
    while training:
        tokens_per_iter = gpt.batch_size * gpt.sample_len
        batch = get_batch_from_stream(openwebtext_ds)
        loss = gpt.train(batch)
        losses.append(loss)
        i += 1
        elapsed = time.time() - start
        stats["iter"] = i
        stats["loss"] = loss
        stats["best_loss"] = min(stats["best_loss"], loss)
        stats["iter_per_sec"] = i / elapsed
        stats["tokens_per_sec"] = (i * tokens_per_iter) / elapsed
        stats["total_tokens"] = i * tokens_per_iter
        stats["elapsed"] = elapsed
        stats["lr"] = gpt.optimizer.param_groups[0]["lr"]
        if i % 100 == 0:
            print(f"{i}: loss={loss:.4f} | {stats['iter_per_sec']:.1f} it/s | {stats['tokens_per_sec']:.0f} tok/s")

thread = threading.Thread(target=train_loop, daemon=True)
thread.start()

def generate(prompt, length):
    return gpt.generate(prompt, int(length))

def get_gpu_mem():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    return "N/A (CPU)"

def refresh_monitor():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (iter {stats['iter']})")

    mins = stats["elapsed"] / 60
    status_text = (
        f"Iteration:    {stats['iter']}\n"
        f"Loss:         {stats['loss']:.4f}\n"
        f"Best Loss:    {stats['best_loss']:.4f}\n"
        f"Speed:        {stats['iter_per_sec']:.1f} it/s\n"
        f"Throughput:   {stats['tokens_per_sec']:.0f} tokens/s\n"
        f"Total Tokens: {stats['total_tokens']:,}\n"
        f"Training Time:{mins:.1f} min\n"
        f"LR:           {stats['lr']}\n"
        f"GPU Memory:   {get_gpu_mem()}"
    )
    return fig, status_text

with gr.Blocks() as app:
    with gr.Tab("Generate"):
        prompt = gr.Textbox(label="Prompt")
        length = gr.Slider(10, 500, value=200, label="Length")
        output = gr.Textbox(label="Output")
        btn = gr.Button("Generate")
        btn.click(generate, inputs=[prompt, length], outputs=output)

    with gr.Tab("Training Monitor"):
        plot = gr.Plot(label="Loss Curve")
        status = gr.Textbox(label="Stats")
        refresh = gr.Button("Refresh")
        refresh.click(refresh_monitor, outputs=[plot, status])

        lr_slider = gr.Slider(0.0001, 0.01, value=3e-4, label="Learning Rate")
        lr_btn = gr.Button("Update LR")

        def update_lr(new_lr):
            for g in gpt.optimizer.param_groups:
                g['lr'] = new_lr
            return f"LR updated to {new_lr}"

        lr_btn.click(update_lr, inputs=[lr_slider], outputs=status)

        bs_slider = gr.Slider(1, 128, value=32, step=1, label="Batch Size")
        bs_btn = gr.Button("Update Batch Size")

        def update_bs(new_bs):
            gpt.batch_size = int(new_bs)
            return f"Batch size updated to {int(new_bs)}"

        bs_btn.click(update_bs, inputs=[bs_slider], outputs=status)

app.launch(server_name="0.0.0.0", share=True)