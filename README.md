MNIST Digit Classification
Bu proje, MNIST veri seti kullanarak el yazısı rakamları sınıflandıran bir yapay zeka modelinin eğitimini ve tahminini gerçekleştirmektedir. Proje, TensorFlow ve Keras kullanarak bir sinir ağı modeli oluşturur ve modelin doğruluğunu test eder.

Proje Hedefi
Bu proje, el yazısıyla yazılmış rakamları sınıflandırmayı hedefleyen bir yapay zeka modelini eğitir. Model, MNIST veri seti üzerinde eğitim alır ve test verileriyle doğruluğunu değerlendirir.

Gerekli Kütüphaneler
Projede kullanılan ana kütüphaneler şunlardır:

tensorflow: Modelin eğitimini ve değerlendirmesini yapar.
numpy: Veri işlemleri için kullanılır.
matplotlib: Eğitim ve test görüntülerini görselleştirir.
Bu kütüphaneleri yüklemek için şu komutu kullanabilirsiniz:


pip install tensorflow numpy matplotlib
Projeyi Çalıştırma
Aşağıdaki adımları takip ederek projeyi çalıştırabilirsiniz.

Gerekli kütüphaneleri yükleyin:
pip install tensorflow numpy matplotlib

Kod dosyasını çalıştırın:
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNIST veri setini yükleme
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi normalize etme (0-255 arasındaki değerleri 0-1 aralığına dönüştürme)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model oluşturma
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme (epoch sayısı 10)
model.fit(x_train, y_train, epochs=10)

# Test verisi üzerinde modelin doğruluğunu değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest doğruluğu: {test_acc}')

# Bir örnek resim tahmini
plt.imshow(x_test[0], cmap=plt.cm.binary)  # Test veri setinden bir görüntü
plt.show()

# Modelin tahminini yapma
predictions = model.predict(x_test)
print(f"İlk test görüntüsüne yapılan tahmin: {np.argmax(predictions[0])}")
Model Yapısı
Model, şu katmanlardan oluşur:

Flatten: 28x28 boyutundaki 2D resimleri tek bir uzun vektöre dönüştürür.
Dense Layer: 128 nörondan oluşan bir tam bağlı katman.
Dropout Layer: Overfitting (aşırı öğrenme) problemini önlemek için %20 dropout oranı.
Dense Output Layer: 10 nöronlu çıkış katmanı, her biri bir rakamı (0-9) temsil eder. Softmax aktivasyon fonksiyonu ile, her bir rakam için olasılık hesaplanır.
Eğitim Süreci
Model 10 epoch boyunca eğitilmiştir ve bu sürede doğruluk oranı artmıştır. Epoch değeri arttırılarak daha tutarlı sonuçlar elde edilebilir.

Kullanıcı Yorumları
Projeyi geliştirirken, farklı eğitim stratejileri ve model yapıları kullanabilirsiniz. Veri augmentasyonu veya modelin eğitim süresini uzatarak modelin doğruluğunu artırmayı düşünebilirsiniz.

Geliştirlebilirlik
Projeyi geliştirmek için web tasarımı ve arayüz eklenebilir.Yeni kodlar eklenerek sayılarla yeni tanışan çocuklar için bir eğitim aracı ve oyun haline getirilebilir.
