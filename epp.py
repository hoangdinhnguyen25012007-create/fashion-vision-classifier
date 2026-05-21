import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fashion_quickdraw.keras")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Đã tải mô hình thành công từ: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"[LỖI] Không tìm thấy file model tại {MODEL_PATH}. Hãy chạy train.py trước!")

LABELS = [
    "Áo thun (T-shirt)", "Quần dài (Pants)", "Váy (Dress)", "Áo khoác (Coat)", "Giày (Shoe)",
    "Dép (Sandal)", "Vớ / Tất (Sock)", "Nón / Mũ (Hat)", "Túi xách (Bag)", "Quần ngắn (Shorts)"
]


def predict_drawing(input_space):
    if input_space is None or "composite" not in input_space:
        return {}
    
    image = input_space["composite"]
    image = image.convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image)
    
    img_array = 255.0 - img_array
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    predictions = model.predict(img_array)[0]
    
    best_index = np.argmax(predictions)
    best_label = LABELS[best_index]
    best_confidence = float(predictions[best_index])
    
    return {best_label: best_confidence}

def clear_canvas():
    return {"composite": None, "background": None, "layers": []}


custom_css = """
body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
}

.premium-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    padding: 32px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
}

.image-container, .image-container var {
    background-color: #f1f5f9 !important;
    border: 2px dashed #3b82f6 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

.image-container {
    height: 420px !important;
}

.button-row {
    margin-top: 16px !important;
    margin-bottom: 24px !important;
    gap: 12px !important;
}

.block .section-header, 
.image-container .sk-label, 
.label-title,
span[data-testid="block-info"] {
    display: none !important;
}

.label-inner {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1d4ed8 !important;
}

.btn-predict {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 46px !important;
    transition: all 0.25s ease !important;
}
.btn-predict:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

.btn-clear {
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 46px !important;
    transition: all 0.25s ease !important;
}
.btn-clear:hover {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
}
"""

with gr.Blocks() as demo:
    
    gr.Markdown(
        """
        <div style='text-align: center; margin-top: 15px; margin-bottom: 30px;'>
            <h1 style='color: #0f172a; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.7px; margin-bottom: 6px;'>
                PROJECT AI NHẬN DIỆN TRANG PHỤC
            </h1>
            <p style='color: #64748b; font-size: 1rem; font-weight: 500;'>
                Nhận diện trang phục từ hình vẽ
            </p>
        </div>
        """
    )
    
    with gr.Column(elem_classes="premium-card"):

        gr.Markdown("<h3 style='color: #0f172a; margin-top: 0; margin-bottom: 12px; font-size: 1.15rem;'>🎨 PHÁC THẢO DÁNG ÁO/QUẦN</h3>")
        
        draw_canvas = gr.Sketchpad(
            type="pil",
            layers=False,        
            transforms=[],       
            show_label=False,    
            brush=gr.Brush(colors=["#000000"], default_size=6)
        )
        
        with gr.Row(elem_classes="button-row"):
            btn_clear = gr.Button("🗑️ Xóa nét vẽ", elem_classes="btn-clear", scale=1)
            btn_predict = gr.Button("🔮 Phân tích hình vẽ", elem_classes="btn-predict", scale=2)
            
        gr.HTML("<hr style='border: 0; border-top: 1px solid #f1f5f9; margin-bottom: 20px;'>")
        
        gr.Markdown("<h3 style='color: #0f172a; margin-top: 0; margin-bottom: 12px; font-size: 1.15rem;'>🔮 KẾT QUẢ PHÂN TÍCH</h3>")
        
        output_labels = gr.Label(num_top_classes=1,show_label=False)

    gr.Markdown(
        """
        <div style='text-align: center; margin-top: 25px;'>
            <p style='color: #94a3b8; font-size: 0.85rem;'>
                💡 Mẹo: Vẽ càng đẹp, độ chính xác càng cao.
            </p>
        </div>
        """
    )
  
    btn_predict.click(fn=predict_drawing,inputs=draw_canvas,outputs=output_labels)

    btn_clear.click(fn=clear_canvas,inputs=None,outputs=draw_canvas)
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(), css=custom_css, share=False)