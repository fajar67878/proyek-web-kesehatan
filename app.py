from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# --- BAGIAN AI (PROSES TRAINING) ---
# Memastikan file csv ada sebelum dibaca
if os.path.exists('data_berat.csv'):
    df = pd.read_csv('data_berat.csv')
    X = df[['kalori']] # Fitur
    y = df['berat_badan'] # Target
    
    # Membuat dan melatih model AI
    model = LinearRegression()
    model.fit(X, y)
else:
    print("❌ ERROR: File data_berat.csv TIDAK DITEMUKAN di folder ini!")
    
# ----------------------------------

@app.route('/prediksi-berat', methods=['GET', 'POST'])
def prediksi_berat():
    hasil = None
    kalori = None
    
    if request.method == 'POST':
        try:
            kalori = float(request.form['kalori'])
            
            # Prediksi menggunakan Model AI
            input_data = np.array([[kalori]])
            prediksi = model.predict(input_data)
            hasil = round(float(prediksi[0]), 2)
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            
    # Ubah bagian return di fungsi prediksi_berat() menjadi seperti ini:
    return render_template('index.html', 
                           hasil=hasil, 
                           kalori=kalori, 
                           data_tabel=df.to_dict(orient='records'))
if __name__ == '__main__':
    print("🚀 Memulai Server Flask...")
    app.run(debug=True)
