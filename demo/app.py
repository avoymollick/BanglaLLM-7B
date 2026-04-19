"""
BanglaLLM Demo — 5-tab Gradio interface
"""
import os, sys, torch, gradio as gr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.bangla_llm import BanglaLLM, Config
from tokenizers import Tokenizer

CKPT = os.environ.get("CHECKPOINT", "checkpoints/7b/best_checkpoint.pt")
TOK  = os.environ.get("TOKENIZER",  "tokenizer/tokenizer.json")
TOK  = TOK if os.path.exists(TOK) else "tokenizer/tokenizer.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {CKPT}...")
print(f"Device: {device}")

ck  = torch.load(CKPT, map_location=device, weights_only=False)
cfg = Config(ck["cfg"]["size"])
model = BanglaLLM(cfg).to(device)
if device.type == "cuda":
    model = model.to(torch.bfloat16)
model.load_state_dict(ck["model"])
model.eval()
print(f"Model loaded! size={cfg.size} | step={ck['step']:,}")

tok = Tokenizer.from_file(TOK)
BOS = tok.token_to_id("<s>") or 1
EOS = tok.token_to_id("</s>") or 2


def generate(prompt, max_new=200, temperature=0.8, top_p=0.9, rep_penalty=1.1):
    if not prompt.strip():
        return "অনুগ্রহ করে কিছু লিখুন।"
    ids = torch.tensor([[BOS] + tok.encode(prompt.strip()).ids]).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new=max_new, temp=temperature,
                              top_p=top_p, rep_pen=rep_penalty, eos=EOS)
    return tok.decode(out[0].tolist(), skip_special_tokens=True)


def complete_text(text, max_tokens, temp, top_p):
    return generate(text, int(max_tokens), temp, top_p)

def answer_question(question, max_tokens, temp):
    prompt = f"প্রশ্ন: {question}\nউত্তর:"
    return generate(prompt, int(max_tokens), temp)

def generate_story(opening, max_tokens, temp):
    prompt = f"গল্প: {opening}"
    return generate(prompt, int(max_tokens), temp)

def generate_news(topic, max_tokens, temp):
    prompt = f"সংবাদ শিরোনাম: {topic}\nবিস্তারিত:"
    return generate(prompt, int(max_tokens), temp)

def summarize(text, max_tokens, temp):
    prompt = f"নিম্নলিখিত পাঠের সারসংক্ষেপ করুন:\n{text}\nসারসংক্ষেপ:"
    return generate(prompt, int(max_tokens), temp)


with gr.Blocks(title="BanglaLLM-7B") as demo:
    gr.Markdown("# 🇧🇩 BanglaLLM-7B\nFirst Native Bengali Large Language Model — Trained From Scratch")
    gr.Markdown(f"**Model:** {cfg.size.upper()} | **Step:** {ck['step']:,} | **Device:** {device}")

    with gr.Tab("📝 টেক্সট কমপ্লিশন"):
        with gr.Row():
            with gr.Column():
                t1 = gr.Textbox(label="বাংলা টেক্সট লিখুন", placeholder="বাংলাদেশ একটি...", lines=3)
                with gr.Row():
                    sl1 = gr.Slider(50, 400, 200, label="সর্বোচ্চ টোকেন")
                    sl2 = gr.Slider(0.1, 1.5, 0.8, label="তাপমাত্রা")
                    sl3 = gr.Slider(0.1, 1.0, 0.9, label="Top-p")
                b1 = gr.Button("তৈরি করুন", variant="primary")
            o1 = gr.Textbox(label="আউটপুট", lines=8)
        b1.click(complete_text, [t1, sl1, sl2, sl3], o1)

    with gr.Tab("❓ প্রশ্নোত্তর"):
        with gr.Row():
            with gr.Column():
                t2 = gr.Textbox(label="প্রশ্ন করুন", placeholder="বাংলাদেশের রাজধানী কী?", lines=2)
                with gr.Row():
                    sl4 = gr.Slider(50, 400, 200, label="সর্বোচ্চ টোকেন")
                    sl5 = gr.Slider(0.1, 1.0, 0.7, label="তাপমাত্রা")
                b2 = gr.Button("উত্তর দিন", variant="primary")
            o2 = gr.Textbox(label="উত্তর", lines=8)
        b2.click(answer_question, [t2, sl4, sl5], o2)

    with gr.Tab("📖 গল্প লেখা"):
        with gr.Row():
            with gr.Column():
                t3 = gr.Textbox(label="গল্পের শুরু", placeholder="একদিন এক রাজার ছেলে...", lines=2)
                with gr.Row():
                    sl6 = gr.Slider(100, 500, 300, label="সর্বোচ্চ টোকেন")
                    sl7 = gr.Slider(0.5, 1.5, 0.9, label="তাপমাত্রা")
                b3 = gr.Button("গল্প লিখুন", variant="primary")
            o3 = gr.Textbox(label="গল্প", lines=10)
        b3.click(generate_story, [t3, sl6, sl7], o3)

    with gr.Tab("📰 সংবাদ"):
        with gr.Row():
            with gr.Column():
                t4 = gr.Textbox(label="সংবাদ বিষয়", placeholder="ঢাকায় আজ...", lines=2)
                with gr.Row():
                    sl8 = gr.Slider(100, 500, 250, label="সর্বোচ্চ টোকেন")
                    sl9 = gr.Slider(0.3, 1.0, 0.7, label="তাপমাত্রা")
                b4 = gr.Button("সংবাদ লিখুন", variant="primary")
            o4 = gr.Textbox(label="সংবাদ", lines=10)
        b4.click(generate_news, [t4, sl8, sl9], o4)

    with gr.Tab("📋 সারসংক্ষেপ"):
        with gr.Row():
            with gr.Column():
                t5 = gr.Textbox(label="বাংলা টেক্সট", placeholder="দীর্ঘ বাংলা অনুচ্ছেদ...", lines=6)
                with gr.Row():
                    sl10 = gr.Slider(50, 300, 150, label="সর্বোচ্চ টোকেন")
                    sl11 = gr.Slider(0.3, 1.0, 0.6, label="তাপমাত্রা")
                b5 = gr.Button("সারসংক্ষেপ করুন", variant="primary")
            o5 = gr.Textbox(label="সারসংক্ষেপ", lines=6)
        b5.click(summarize, [t5, sl10, sl11], o5)

demo.launch(share=True, server_port=7860)
