import gradio as gr
from model import LLM
import os
import threading
import urllib.request

if not os.path.exists("shakespeare.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt", "r") as f:
    corpus = f.read()

gpt = LLM(corpus, batch_size=512, sample_len=32, d_model=64, d_k=16, n_layers=4, lr=1e-3)

losses = []
training = True

def train_loop():
    global training
    i = 0
    while training:
        loss = gpt.train()
        losses.append(loss)
        i += 1
        if i % 100 == 0:
            print(f"{i}: loss={loss:.4f}")

thread = threading.Thread(target=train_loop, daemon=True)
thread.start()

def generate(prompt, length):
    return gpt.generate(prompt, int(length))

def get_loss_plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (iter {len(losses)})")
    return fig

with gr.Blocks() as app:
    with gr.Tab("Generate"):
        prompt = gr.Textbox(label="Prompt")
        length = gr.Slider(10, 500, value=200, label="Length")
        output = gr.Textbox(label="Output")
        btn = gr.Button("Generate")
        btn.click(generate, inputs=[prompt, length], outputs=output)

    with gr.Tab("Training Monitor"):
        plot = gr.Plot(label="Loss Curve")
        refresh = gr.Button("Refresh")
        refresh.click(get_loss_plot, outputs=plot)

app.launch(server_name="0.0.0.0")


# import gradio as gr
# from model import LLM
# import os

# if not os.path.exists("shakespeare.txt"):
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     urllib.request.urlretrieve(url, "shakespeare.txt")

# with open("shakespeare.txt", "r") as f:
#     corpus = f.read()

# gpt = LLM(corpus, batch_size=512, sample_len=32, d_model=64, d_k=16, n_layers=4, lr=1e-3)

# for i in range(10000):
#     loss = gpt.train()
#     if i % 100 == 0:
#         print(f"{i}: loss={loss:.4f}")

# def generate(prompt, length):
#     return gpt.generate(prompt, int(length))

# gr.Interface(
#     fn=generate,
#     inputs=["text", gr.Slider(10, 500, value=200)],
#         outputs="text"
# ).launch(server_name="0.0.0.0")