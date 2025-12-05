# CSE473s Fall-2025 – Build-Your-Own Neural-Network Library

---

## ✅ What Works Today

| Milestone | Status | Evidence |
|-----------|--------|----------|
| **Core library** | ✅ | `/lib` – modular, vectorised, zero external-ML imports |
| **XOR problem** | ✅ | 2-4-1 MLP reaches **0% classification error** in &lt;1k epochs |
| **Gradient check** | ✅ | numerical vs analytical **&lt;1e-7** |

---

## 📦 Repo Structure
```
├── lib/               # our tiny framework
│   ├── layers.py      # Dense + base Layer
│   ├── activations.py # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py      # MSE
│   ├── optimizers.py  # SGD
│   └── network.py     # Sequential container
├── notebooks/
│   └── project_demo.ipynb  # all runnable demos
└── README.md
```

---

## 🚪 Next Gates (in order)

1. **Auto-encoder on MNIST**  
   - 784 → 32 → 784, ReLU / Sigmoid, MSE loss  
   - Deliver loss-curve + original-vs-reconstructed grid

2. **Latent-SVM classifier**  
   - Freeze encoder, train `sklearn.svm.SVC` on 32-D features  
   - Report test accuracy + confusion matrix

3. **TensorFlow/Keras baseline**  
   - Replicate **exact** architectures (XOR & auto-encoder)  
   - Compare LOC, training time, final MSE, test acc

4. **Final report** (`report/project_report.pdf`)  
   - Design choices, results, comparison, lessons learnt
