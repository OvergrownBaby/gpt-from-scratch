import gradio as gr
from model import LLM
import os
import threading
import urllib.request
import time
import torch

if not os.path.exists("shakespeare.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt", "r") as f:
    corpus = f.read()

gpt = LLM(corpus, batch_size=512, sample_len=32, d_model=64, d_k=16, n_layers=4, lr=1e-3)

losses = []
training = True

stats = {"iter": 0, "loss": 0, "best_loss": float("inf"), "iter_per_sec": 0, "tokens_per_sec": 0, "total_tokens": 0, "elapsed": 0, "lr": 1e-3}

def train_loop():
    global training
    i = 0
    tokens_per_iter = gpt.batch_size * gpt.sample_len
    start = time.time()
    while training:
        loss = gpt.train()
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

app.launch(server_name="0.0.0.0", share=True)