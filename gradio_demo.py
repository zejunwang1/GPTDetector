# coding=utf-8

import argparse
import gradio as gr
import os
import torch
from easytokenizer import AutoTokenizer
from transformers import AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--han_ratio", type=float, default=0.4)
    parser.add_argument("--ai_ratio", type=float, default=0.5)
    parser.add_argument("--human_ratio", type=float, default=0.3)
    args = parser.parse_args()
    return args

args = parse_args()

# Load model and tokenizer
device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')
model = AutoModelForSequenceClassification.from_pretrained(args.model)
model.to(device)
model = model.eval()

vocab_path = os.path.join(args.model, 'vocab.txt')
assert os.path.exists(vocab_path)
tokenizer = AutoTokenizer(vocab_path, do_lower_case=True)

def hanRatio(text):
    n = 0
    for ch in text:
        cp = ord(ch)
        if cp >= 0x4E00 and cp <= 0x9FA5:
            n += 1
    return n / len(text)

def text_splitter(text, max_length):
    n = len(text)
    beg, cur = 0, 0
    chunk_size = 0
    chunks, locs = [], []
    backtracking = int(max_length / 2)
    while cur < n:
        cur += 1
        chunk_size += 1
        if chunk_size == max_length:
            move = 1
            while move < backtracking:
                ch = text[cur - move]
                if ch == '\n':
                    cur = cur - move + 1
                    break
                move += 1
            if move == backtracking:
                move = 1
                while move < backtracking:
                    ch = text[cur - move]
                    if ch in ['。', '！', '？', '；', '!', '?', ';']:
                        cur = cur - move + 1
                        break
                    move += 1
                if move == backtracking:
                    move = 1
                    while move < backtracking:
                        ch = text[cur - move]
                        if ch in ['，', ',', '、', '\t', ' ']:
                            cur = cur - move + 1
                            break
                        move += 1

            chunk = text[beg : cur]
            if hanRatio(chunk) >= args.han_ratio:
                chunks.append(chunk)
                locs.append(beg)
            beg = cur
            chunk_size = 0

    if chunk_size:
        chunk = text[beg : ]
        if hanRatio(chunk) >= args.han_ratio:
            if not chunks or len(chunk) >= backtracking:
                chunks.append(chunk)
                locs.append(beg)
            else:
                chunks[-1] += chunk
    return (chunks, locs)

def detect(text, chunk_size, p_threshold):
    ai_locs = []
    ai_num_chars = 0
    chunks, locs = text_splitter(text, chunk_size)
    n = len(chunks)
    if n == 0:
        return "", ""
    num_batches = int((n - 1) / args.batch_size) + 1
    for i in range(num_batches):
        start = i * args.batch_size
        end = min((i + 1) * args.batch_size, n)
        batch_chunks = chunks[start : end]
        batch_encodings = tokenizer.encode(batch_chunks, add_cls_sep=False, truncation=True, max_length=1024)
        batch_input_ids = torch.tensor(batch_encodings["input_ids"], device=device)
        batch_attention_mask = torch.tensor(batch_encodings["attention_mask"], device=device)
        # inference
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs[0]
        probs = logits.softmax(dim=-1)[:, -1]
        for j in range(len(probs)):
            if probs[j] > p_threshold:
                l = len(batch_chunks[j])
                ai_num_chars += l
                ai_locs.append((locs[j + start], l))

    ai_ratio = ai_num_chars / len(text)
    if ai_ratio > args.ai_ratio:
        conclusion = "该文本可能为AI生成的！"
    elif ai_ratio > args.human_ratio:
        conclusion = "该文本可能部分为AI生成的！"
    else:
        conclusion = "该文本为人类书写的！"

    if ai_ratio == 0 or ai_ratio == 1:
        return conclusion, ""
    chars = list(text)
    for a, b in ai_locs:
        chars[a] = "<font style='color:red'>" + chars[a]
        chars[a + b - 1] += "</font>"
    detail = ''.join(chars)
    return conclusion, detail.replace('\n', '<br>')

with gr.Blocks() as demo:
    gr.HTML("""<h2 align="center">AI生成文本检测系统</h2>""")
    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=14, container=False)
    with gr.Row():
        with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
            chunk_size = gr.Slider(256, 1024, value=768, step=1, label="Chunk size", interactive=True)
            p_threshold = gr.Slider(0, 1, value=0.7, step=0.01, label="AI probability", interactive=True)

    submit_button = gr.Button("Submit", variant="primary")
    conclusion = gr.Textbox(label="Conclusion:")
    detail = gr.HTML(label="Details:")
    submit_button.click(detect, [user_input, chunk_size, p_threshold], [conclusion, detail])

demo.launch(server_name="0.0.0.0", server_port=args.port)

