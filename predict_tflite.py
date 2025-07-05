import numpy as np
from PIL import Image
import tensorflow as tf
import os

# تحميل النموذج
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# معلومات الإدخال والإخراج
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# تجهيز الصورة
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB').resize((224, 224))
    except FileNotFoundError:
        print(f"❌ الملف '{image_path}' غير موجود. حاولي مرة ثانية.")
        return None
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# توقع
def predict(image_path):
    image = preprocess_image(image_path)
    if image is None:
        return
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]

    # قراءة أسماء الفئات
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    print(f"✅ النتيجة: {labels[predicted_index]} (الثقة: {confidence:.2f})")

# تشغيل السكربت
if __name__ == "__main__":
    while True:
        image_name = input("📸 أدخلي اسم الصورة (أو اكتبي 'خروج' لإنهاء البرنامج): ").strip()
        if image_name.lower() == "خروج":
            print("👋 تم إنهاء البرنامج.")
            break
        predict(image_name)
