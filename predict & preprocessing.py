# Menentukan dimensi input yang diperlukan oleh model
input_shape = (224, 224, 3)  # Ganti sesuai dengan model Anda

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil gambar dari permintaan POST
    image = request.files['image'].read()

    # Preprocess gambar
    image = preprocess_image(image, input_shape)

    # Inferensi menggunakan model TensorFlow Lite
    input_tensor_index = model.get_input_details()[0]['index']
    output = model.tensor(model.get_output_details()[0]['index'])

    # Ubah gambar ke bentuk array numpy
    input_data = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    model.set_tensor(input_tensor_index, input_data)
    model.invoke()

    # Ambil hasil prediksi
    prediction = output()[0]

    # Kirim hasil prediksi sebagai JSON
    return jsonify({'prediction': prediction.tolist()})

def preprocess_image(image, input_shape):
    # Resize gambar ke dimensi input model
    img = Image.open(io.BytesIO(image))
    img = img.resize((input_shape[0], input_shape[1]))

    # Konversi gambar ke array numpy, normalisasi nilai piksel (0-1)
    img_array = np.array(img) / 255.0

    # Menambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
