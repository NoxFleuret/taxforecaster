MODEL_INFO = {
    "Prophet": {
        "description": "Dikembangkan oleh Facebook, Prophet dirancang untuk meramal data time series yang memiliki pola musiman kuat dan tidak beraturan.",
        "mechanism": "Menggunakan model aditif (penjumlahan) yang mengurai tren non-linear, komponen musiman (tahunan, mingguan, harian), dan efek hari libur.",
        "suitable_for": "Data dengan tren jangka panjang, musiman yang kuat, dan data yang memiliki outlier atau hari libur (misal: efek lebaran/akhir tahun).",
        "pros": ["Tahan terhadap data hilang/outlier", "Otomatis mendeteksi perubahan tren", "Intuitif parameter-nya"],
        "cons": ["Kurang akurat untuk jangka pendek yang sangat fluktuatif", "Tidak sefleksibel model Machine Learning untuk fitur eksternal kompleks"]
    },
    "SARIMA": {
        "description": "Seasonal AutoRegressive Integrated Moving Average. Standar emas statistik klasik untuk data musiman.",
        "mechanism": "Mengkombinasikan Autoregresi (AR - nilai masa lalu), Integrasi (I - differencing untuk stasioneritas), dan Moving Average (MA - error masa lalu), ditambah komponen musiman (Seasonal).",
        "suitable_for": "Data time series univariat (hanya satu variabel) dengan pola musiman yang stabil dan jelas.",
        "pros": ["Sangat kuat untuk menangkap siklus linear", "Teori statistik yang mapan dan teruji"],
        "cons": ["Berat secara komputasi untuk data besar", "Sensitif terhadap outlier", "Asumsi linearitas yang kaku"]
    },
    "SARIMAX": {
        "description": "Perluasan dari SARIMA yang mendukung variabel eksternal (eXogenous factors).",
        "mechanism": "Sama seperti SARIMA, namun menambahkan regresi linear variabel eksternal (seperti Inflasi, GDP) untuk membantu prediksi.",
        "suitable_for": "Forecasting yang sangat dipengaruhi faktor ekonomi makro (contoh: PPN sangat tergantung GDP).",
        "pros": ["Bisa memodelkan dampak kebijakan ekonomi", "Lebih akurat jika variabel eksternal punya korelasi kuat"],
        "cons": ["Membutuhkan data masa depan dari variabel eksternal (forecast makro harus ada dulu)"]
    },
    "Holt-Winters": {
        "description": "Juga dikenal sebagai Triple Exponential Smoothing.",
        "mechanism": "Melakukan smoothing (penghalusan) data dengan memberikan bobot lebih besar pada data terbaru. Memiliki 3 komponen: Level (rata-rata), Trend (kemiringan), dan Seasonal (musim).",
        "suitable_for": "Data dengan tren dan musiman yang sangat teratur/repetitif.",
        "pros": ["Sangat cepat dan ringan", "Mudah diinterpretasi"],
        "cons": ["Tidak bisa menangkap interaksi kompleks antar variabel", "Buruk untuk prediksi jangka panjang jika tren berubah drastis"]
    },
    "Theta Model": {
        "description": "Metode Theta adalah model statistik yang sederhana namun sangat efektif. Pemenang kompetisi forecasting M3.",
        "mechanism": "Mendekomposisi data berdasarkan kelengkungan kedua dari tren (Theta lines). Menggabungkan tren jangka panjang dan fluktuasi jangka pendek.",
        "suitable_for": "Data univariat yang tidak memiliki pola musiman terlalu kompleks.",
        "pros": ["Sering mengalahkan model kompleks (benchmark kuat)", "Sangat stabil"],
        "cons": ["Minim parameter yang bisa diandalkan untuk kustomisasi"]
    },
    "XGBoost": {
        "description": "eXtreme Gradient Boosting. Algoritma Machine Learning berbasis pohon keputusan (Decision Trees) yang sangat populer.",
        "mechanism": "Membangun ratusan pohon keputusan secara berurutan. Setiap pohon baru mencoba memperbaiki error dari pohon sebelumnya (Boosting).",
        "suitable_for": "Data tabular yang kompleks dengan hubungan non-linear. Sangat bagus jika kita punya banyak fitur tambahan (makro ekonomi).",
        "pros": ["Seringkali memberikan akurasi tertinggi", "Bisa menangkap pola non-linear yang rumit", "Mendukung Feature Importance"],
        "cons": ["Black-box (sulit dijelaskan secara manual)", "Rawan overfitting jika data sedikit"]
    },
    "LightGBM": {
        "description": "Light Gradient Boosting Machine. Varian boosting yang dikembangkan Microsoft, fokus pada kecepatan.",
        "mechanism": "Mirip XGBoost tapi menggunakan teknik pertumbuhan pohon 'leaf-wise' yang lebih efisien memori dan cepat.",
        "suitable_for": "Data set yang sangat besar.",
        "pros": ["Jauh lebih cepat dari XGBoost", "Hemat memori"],
        "cons": ["Bisa kurang stabil pada dataset kecil (<10.000 data)"]
    },
    "CatBoost": {
        "description": "Categorical Boosting. Dikembangkan oleh Yandex, terkenal karena penanganan data kategorikal yang superior.",
        "mechanism": "Menggunakan Symmetric Trees yang berbeda dengan XGBoost/LightGBM, membuatnya lebih seimbang dan minim overfitting.",
        "suitable_for": "Data finansial yang mungkin memiliki fitur kategori atau data yang butuh model robust.",
        "pros": ["Sangat akurat 'out-of-the-box' tanpa banyak tuning", "Tahan overfitting"],
        "cons": ["Training bisa lebih lambat dari LightGBM"]
    },
    "Random Forest": {
        "description": "Ensemble learning yang menggunakan metode Bagging (Bootstrap Aggregating).",
        "mechanism": "Membuat banyak pohon keputusan secara paralel (bukan berurutan) dengan data acak, lalu mengambil rata-rata (voting) dari semua pohon tersebut.",
        "suitable_for": "Data yang noisy (banyak gangguan) dan beresiko overfitting.",
        "pros": ["Sangat stabil dan tahan banting (robust)", "Jarang overfitting dibanding Boosting", "Mudah disetting"],
        "cons": ["Ukuran model bisa besar (banyak pohon)", "Tidak seakurat XGBoost untuk pola yang sangat halus"]
    },
    "Gradient Boosting": {
        "description": "Versi fundamental dari teknik Boosting (sebelum XGBoost/LightGBM).",
        "mechanism": "Secara iteratif menambahkan model yang memprediksi sisa error (residual) dari model gabungan sebelumnya.",
        "suitable_for": "Dataset ukuran memengah dengan pola non-linear.",
        "pros": ["Fondasi statistik yang kuat", "Mendukung berbagai loss function"],
        "cons": ["Lebih lambat dilatih dibanding XGBoost/LightGBM"]
    },
    "Extra Trees": {
        "description": "Extremely Randomized Trees. Varian dari Random Forest yang lebih 'acak'.",
        "mechanism": "Mirip Random Forest tapi pemilihan fitur dan nilai split-nya dilakukan lebih acak lagi untuk mengurangi variansi.",
        "suitable_for": "Data dengan noise tinggi atau variasi yang sangat liar.",
        "pros": ["Bisa lebih cepat dari Random Forest", "Kadang lebih akurat untuk data yang sangat fluktuatif"],
        "cons": ["Bias lebih tinggi (kurang presisi untuk pola deterministik)"]
    },
    "AdaBoost": {
        "description": "Adaptive Boosting. Salah satu algoritma boosting pertama.",
        "mechanism": "Metode boosting yang memberi bobot lebih pada data yang 'salah diprediksi' oleh model sebelumnya, memaksa model berikutnya untuk fokus pada kasus-kasus sulit tersebut.",
        "suitable_for": "Data yang bersih tanpa banyak outlier.",
        "pros": ["Sederhana dan seringkali efektif", "Sedikit parameter untuk dituning"],
        "cons": ["Sangat sensitif terhadap outlier dan noise"]
    },
    "KNN": {
        "description": "K-Nearest Neighbors. Algoritma berbasis instansi (Instance-based learning).",
        "mechanism": "Memprediksi nilai berdasarkan rata-rata dari 'K' tetangga terdekat dalam ruang fitur (berdasarkan kemiripan pola data).",
        "suitable_for": "Pola data yang berulang atau memiliki klaster tertentu.",
        "pros": ["Sederhana (Lazy Learning)", "Non-parametrik (tidak berasumsi distribusi data)"],
        "cons": ["Menjadi lambat jika data sangat besar", "Performa buruk jika dimensi sangat tinggi (Curse of Dimensionality)"]
    },
    "Neural Network (MLP)": {
        "description": "Multi-Layer Perceptron. Jaringan saraf tiruan sederhana.",
        "mechanism": "Terdiri dari lapisan input, lapisan tersembunyi (hidden layers) dengan fungsi aktivasi, dan output. Belajar pola kompleks melalui backpropagation.",
        "suitable_for": "Hubungan non-linear yang sangat kompleks dan tidak bisa ditangkap model pohon/linear.",
        "pros": ["Bisa memodelkan fungsi yang sangat rumit", "Dasar dari Deep Learning"],
        "cons": ["Butuh data banyak", "Sulit diinterpretasi (Blackbox)", "Butuh scaling data yang baik"]
    },
    "SVR": {
        "description": "Support Vector Regression. Adaptasi dari SVM untuk kasus regresi.",
        "mechanism": "Mencari 'hyperplane' terbaik dalam ruang dimensi tinggi yang memuat sebanyak mungkin poin data dalam batas toleransi error (margin).",
        "suitable_for": "Dataset kecil hingga menengah dengan dimensi tinggi (banyak fitur).",
        "pros": ["Sangat efektif di dimensi tinggi", "Tahan outlier (robust)"],
        "cons": ["Sangat berat untuk dataset besar", "Butuh scaling data yang ketat"]
    },
    "Ridge": {
        "description": "Regresi Linear dengan regularisasi L2.",
        "mechanism": "Sama seperti regresi linear biasa (garis lurus), tapi menambahkan denda (penalty) pada koefisien agar tidak terlalu besar.",
        "suitable_for": "Data yang memiliki multikolinearitas (fitur yang mirip-mirip/berhubungan erat).",
        "pros": ["Mencegah overfitting pada model linear", "Solusi stabil"],
        "cons": ["Hanya bisa menangkap hubungan linear (garis lurus)"]
    },
    "ElasticNet": {
        "description": "Kombinasi antara Ridge dan Lasso.",
        "mechanism": "Menggabungkan penalti L1 (Lasso) dan L2 (Ridge). Bisa mengecilkan koefisien (Ridge) sekaligus membuang fitur yang tidak berguna (Lasso).",
        "suitable_for": "Banyak fitur yang berkorelasi satu sama lain.",
        "pros": ["Fleksibel", "Bisa melakukan seleksi fitur otomatis"],
        "cons": ["Perlu tuning dua parameter (alpha dan l1_ratio)"]
    },
    "Lasso": {
        "description": "Least Absolute Shrinkage and Selection Operator.",
        "mechanism": "Regresi linear dengan penalti L1. Penalti ini bisa memaksa koefisien fitur yang kurang penting menjadi tepat NOL.",
        "suitable_for": "Seleksi fitur (feature selection) secara otomatis.",
        "pros": ["Menghasilkan model yang 'sparse' (sederhana)", "Membuang fitur sampah"],
        "cons": ["Jika dua fitur berkorelasi tinggi, salah satu akan dibuang secara acak"]
    },
    "Bayesian Ridge": {
        "description": "Regresi Linear dengan pendekatan Probabilistik Bayesian.",
        "mechanism": "Mengestimasi distribusi koefisien regresi daripada nilai tetap. Otomatis menyeimbangkan regularisasi.",
        "suitable_for": "Data kecil di mana kita butuh ketahanan terhadap overfitting.",
        "pros": ["Tidak perlu tuning parameter regularisasi manual", "Memberikan interval kepercayaan (probabilistik)"],
        "cons": ["Lambat untuk data dimensi besar"]
    },
    "Gaussian Process": {
        "description": "Gaussian Process Regression (GPR). Metode non-parametrik berbasis kernel.",
        "mechanism": "Memprediksi distribusi probabilitas untuk setiap titik, bukan hanya satu nilai. Menggunakan kernel untuk mengukur kemiripan antar data.",
        "suitable_for": "Dataset kecil (<1000 titik) yang sangat non-linear dan butuh estimasi ketidakpastian (confidence intervals).",
        "pros": ["Menangkap ketidakpastian prediksi dengan sangat baik", "Sangat fleksibel (non-linear)"],
        "cons": ["Sangat lambat (kompleksitas kubik)", "Boros memori untuk data besar"]
    },
    "Huber Regressor": {
        "description": "Robust Linear Regression.",
        "mechanism": "Model linear yang tidak sensitif terhadap outlier. Menggunakan Huber Loss (kombinasi squared loss dan absolute loss).",
        "suitable_for": "Data yang mengandung banyak pencilan (outlier) atau error ekstrim.",
        "pros": ["Sangat tahan terhadap data 'kotor'", "Lebih stabil daripada OLS biasa"],
        "cons": ["Komputasi sedikit lebih berat daripada regresi linear simpel"]
    },
    "Kernel Ridge": {
        "description": "Ridge Regression dengan Kernel Trick.",
        "mechanism": "Menggabungkan Regresi Ridge dengan Kernel (seperti pada SVM) untuk menangkap pola non-linear.",
        "suitable_for": "Pola non-linear halus dengan datasets ukuran menengah.",
        "pros": ["Closed-form solution (matematis pasti)", "Bisa menangkap pola kompleks"],
        "cons": ["Memori intensif (menyimpan matriks kernel nxn)"]
    },
    "Polynomial Regression": {
        "description": "Regresi polinomial (kurva).",
        "mechanism": "Mengubah fitur input menjadi pangkat-pangkatnya (x, x^2, x^3) lalu melakukan regresi linear. Ini memungkinkan model membuat garis lengkung.",
        "suitable_for": "Tren data yang jelas melengkung (misal pertumbuhan eksponensial di awal lalu melandai).",
        "pros": ["Sangat sederhana namun powerful untuk kurva", "Interpretasi mudah"],
        "cons": ["Rawan osilasi liar (Runge phenomenon) pada derajat tinggi"]
    }
}
