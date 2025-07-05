# Task 1: Teachable Machine Image Classification

هذا المشروع يستخدم نموذج مدرب بواسطة **Teachable Machine** من جوجل لتصنيف الصور إلى فئتين:  
- Ronaldo  
- Messi

---

## الملفات في المشروع

- `model_unquant.tflite` : نموذج TensorFlow Lite المدرب.  
- `labels.txt` : أسماء الفئات (اللاعبين).  
- `predict_tflite.py` : سكربت بايثون لتحميل النموذج وعمل التنبؤ على صورة مدخلة.  
- `test.jpg`، `test2.jpg` : صور للاختبار (اختياري).

---

## طريقة الاستخدام

1. فعل البيئة الافتراضية (إذا استخدمت pyenv أو أي أداة أخرى).

2. شغل السكربت عبر التيرمنال:

```bash
python predict_tflite.py
3.البرنامج راح يطلب منك اسم الصورة. اكتبي اسم الصورة مع الامتداد (مثلاً test.jpg).
4.راح يعرض لك التوقع مع نسبة الثقة.
