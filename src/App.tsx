import EmbeddingsSection from "@/sections/EmbeddingsSection";
import MatrixSection from "@/sections/MatrixSection";
import AttentionSection from "@/sections/AttentionSection";
import GradientSection from "@/sections/GradientSection";

function Hero() {
  return (
    <header className="relative overflow-hidden border-b border-zinc-800">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(52,211,153,0.08),transparent_55%)]" />
      <div className="mx-auto max-w-6xl px-6 py-28 text-center">
        <p className="font-mono text-sm tracking-widest text-emerald-400">
          LINEAR ALGEBRA × AI
        </p>
        <h1 className="mx-auto mt-4 max-w-3xl text-4xl font-bold leading-tight sm:text-5xl">
          AI is just{" "}
          <span className="bg-gradient-to-r from-emerald-400 via-sky-400 to-pink-400 bg-clip-text text-transparent">
            linear algebra
          </span>
          {" "}you can see
        </h1>
        <p className="mx-auto mt-6 max-w-xl text-lg text-zinc-400">
          Every answer an AI gives is vectors, dot products, matrix multiplies
          and gradients. Explore them hands-on — all math computed live in your
          browser.
        </p>
        <nav className="mt-10 flex flex-wrap justify-center gap-3 font-mono text-xs">
          {[
            ["01 vectors", "#embeddings", "text-emerald-300 border-emerald-800"],
            ["02 matrices", "#matrices", "text-sky-300 border-sky-800"],
            ["03 attention", "#attention", "text-pink-300 border-pink-800"],
            ["04 learning", "#training", "text-amber-300 border-amber-800"],
          ].map(([label, href, cls]) => (
            <a key={href} href={href} className={`rounded-full border px-4 py-2 hover:bg-zinc-900 ${cls}`}>
              {label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="border-t border-zinc-800 py-10 text-center text-sm text-zinc-500">
      <p>
        Built with React + TypeScript · all visualizations run on real math in
        your browser ·{" "}
        <a
          href="https://github.com/GiovanniHiga/linear-algebra-of-ai"
          className="text-zinc-300 underline decoration-zinc-600 hover:text-emerald-400"
        >
          source on GitHub
        </a>
      </p>
    </footer>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
      <Hero />
      <EmbeddingsSection />
      <MatrixSection />
      <AttentionSection />
      <GradientSection />
      <Footer />
    </div>
  );
}
