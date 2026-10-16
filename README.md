# Linear Algebra of AI

An interactive, browser-based visualization of the mathematics that powers modern AI — vectors, matrices, attention and gradient descent. **Every number you see is computed live with real linear algebra in your browser.**

🔗 **[Live demo](https://giovannihiga.github.io/linear-algebra-of-ai/)** · Built with React + TypeScript + Vite + Tailwind CSS

## What's inside

| # | Section | Math shown |
|---|---------|------------|
| 01 | **Words become vectors** | Hand-crafted 8D word embeddings projected to 2D with PCA (power iteration), cosine similarity ranking |
| 02 | **Matrices transform space** | Animated interpolation from the identity to rotations, shears, scales and reflections; y = Wx as geometry |
| 03 | **Attention** | A live, editable attention heatmap: softmax(QKᵀ/√d) over your own sentence |
| 04 | **Learning** | Real gradient descent fitting y = wx + b on data, with the descent path drawn over the loss landscape |

## Run it locally

```bash
npm install
npm run dev        # http://localhost:3000
```

Build for production:

```bash
npm run build      # → dist/ (plain static site)
```

## Project structure

```
src/
├── lib/linalg.ts           # dot, matVec, softmax, PCA… the math core
├── sections/
│   ├── EmbeddingsSection.tsx   # 01 vectors
│   ├── MatrixSection.tsx       # 02 matrices
│   ├── AttentionSection.tsx    # 03 attention
│   └── GradientSection.tsx     # 04 learning
└── App.tsx
```

## Roadmap

- [ ] 3D embedding explorer (Three.js)
- [ ] Animated matrix × matrix multiplication
- [ ] Multi-head attention visualization
- [ ] A real neural network trained in-browser on a 2D classification task

## License

MIT
