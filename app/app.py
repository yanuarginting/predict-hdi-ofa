# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = load('model_ipm.model')

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk variabel input atau features (X) ke model
	input_harapan_lama_sekolah = 10
	input_pengeluaran_perkapita  = 20000
	input_rerata_lama_sekolah = 5
	input_usia_harapan_hidup  = 50
	
	if request.method=='POST':
		# Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
		input_harapan_lama_sekolah = float(request.form['harapan_lama_sekolah'])
		input_pengeluaran_perkapita  = float(request.form['pengeluaran_perkapita'])
		input_rerata_lama_sekolah = float(request.form['rerata_lama_sekolah'])
		input_usia_harapan_hidup  = float(request.form['usia_harapan_hidup'])
		
		# Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
		df_test = pd.DataFrame(data={
			'harapan_lama_sekolah' : [input_harapan_lama_sekolah],
			'pengeluaran_perkapita'  : [input_pengeluaran_perkapita],
			'rerata_lama_sekolah' : [input_rerata_lama_sekolah],
			'usia_harapan_hidup'  : [input_usia_harapan_hidup]
		})

		hasil_prediksi = model.predict(df_test[0:1])[0]

		# Set Path untuk gambar hasil prediksi
		if hasil_prediksi == 'High':
			gambar_prediksi = '/static/images/high.png'
		elif hasil_prediksi == 'Normal':
			gambar_prediksi = '/static/images/high.png'
		elif hasil_prediksi == 'Very-High':
			gambar_prediksi = '/static/images/high.png'
		else:
			gambar_prediksi = '/static/images/low.png'
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"prediksi": hasil_prediksi,
			"gambar_prediksi" : gambar_prediksi
		})

# =[Main]========================================

if __name__ == '__main__':
	
	# # Load model yang telah ditraining
	# model = load('model_ipm.model')

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)
	
	


