/**
 * pages/Dashboard.jsx
 * --------------------
 * Main application page — prediction form + results + history.
 */

import { useEffect, useState } from "react";
import PredictionForm   from "../components/PredictionForm";
import PredictionResult from "../components/PredictionResult";
import HistoryTable     from "../components/HistoryTable";
import { LoadingSpinner, ErrorAlert, StatusBadge } from "../components/ui";
import { usePrediction }  from "../hooks/usePrediction";
import { getHealth, trainModels } from "../services/api";

export default function Dashboard() {
  const { prediction, isLoading, error, predict, reset } = usePrediction();

  const [health,        setHealth]        = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [training,      setTraining]      = useState(false);
  const [trainMsg,      setTrainMsg]      = useState(null);
  const [historyKey,    setHistoryKey]    = useState(0); // bump to refresh table

  // Poll health once on mount
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "error", models_ready: false }))
      .finally(() => setHealthLoading(false));
  }, []);

  async function handleTrain() {
    setTraining(true);
    setTrainMsg(null);
    try {
      const res = await trainModels();
      setTrainMsg({ ok: true, text: res.message });
      // Refresh health status
      const h = await getHealth();
      setHealth(h);
    } catch (e) {
      setTrainMsg({ ok: false, text: e.message });
    } finally {
      setTraining(false);
    }
  }

  async function handlePredict(formData) {
    await predict(formData);
    // Refresh history table after successful prediction
    setHistoryKey((k) => k + 1);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/30 to-violet-50/20">
      {/* ── Top nav ──────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-slate-200/70">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-lg flex items-center justify-center text-white text-sm font-bold">
              T
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-800 leading-none">
                Transport Demand Predictor
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">Powered by Machine Learning</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {healthLoading ? (
              <span className="text-xs text-slate-400">Checking API…</span>
            ) : (
              <StatusBadge ready={health?.models_ready} />
            )}

            <button
              onClick={handleTrain}
              disabled={training}
              className="px-3 py-1.5 text-xs font-medium rounded-lg border border-indigo-200
                bg-indigo-50 text-indigo-700 hover:bg-indigo-100 disabled:opacity-50
                transition flex items-center gap-1.5"
            >
              {training ? (
                <>
                  <span className="w-3 h-3 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
                  Training…
                </>
              ) : "⚡ Train Models"}
            </button>
          </div>
        </div>
      </header>

      {/* ── Hero banner ──────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 pb-4">
        <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-2xl px-8 py-6 text-white">
          <p className="text-indigo-200 text-xs font-medium uppercase tracking-widest mb-2">
            ML-Powered Analytics
          </p>
          <h2 className="text-2xl font-bold">
            Predict Transport Demand
          </h2>
          <p className="text-indigo-200 text-sm mt-1 max-w-lg">
            Enter a date, time, location, and weather to get an instant demand
            forecast using Random Forest or Linear Regression.
          </p>
        </div>
      </section>

      {/* ── Training message ─────────────────────────────────── */}
      {trainMsg && (
        <div className="max-w-6xl mx-auto px-4 sm:px-6 pb-2">
          <div className={`text-sm px-4 py-3 rounded-xl border ${
            trainMsg.ok
              ? "bg-emerald-50 border-emerald-200 text-emerald-700"
              : "bg-red-50 border-red-200 text-red-700"
          }`}>
            {trainMsg.ok ? "✅" : "❌"} {trainMsg.text}
          </div>
        </div>
      )}

      {/* ── Main content ─────────────────────────────────────── */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 pb-12 grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Left — Form */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
          <div className="flex items-center gap-2 mb-5">
            <span className="text-lg">🔮</span>
            <h3 className="font-semibold text-slate-800">Prediction Inputs</h3>
          </div>

          {health?.status === "error" && (
            <ErrorAlert
              message="Cannot reach the backend API. Make sure it's running on port 8000."
              onDismiss={() => setHealth(null)}
            />
          )}

          <PredictionForm onSubmit={handlePredict} isLoading={isLoading} />
        </div>

        {/* Right — Results */}
        <div className="space-y-4">
          {isLoading && (
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
              <LoadingSpinner label="Running ML model…" />
            </div>
          )}

          {error && !isLoading && (
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
              <ErrorAlert message={error} onDismiss={reset} />
            </div>
          )}

          {prediction && !isLoading && (
            <PredictionResult result={prediction} />
          )}

          {!prediction && !isLoading && !error && (
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-8 flex flex-col items-center justify-center text-center gap-3 min-h-[200px]">
              <div className="text-4xl">📊</div>
              <p className="text-slate-600 font-medium">Your prediction will appear here</p>
              <p className="text-slate-400 text-sm">Fill in the form and click Predict Demand</p>
            </div>
          )}
        </div>

        {/* Full-width history */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-lg">🗂️</span>
            <h3 className="font-semibold text-slate-800">Recent Predictions</h3>
          </div>
          {/* key trick: re-mounts HistoryTable after each prediction */}
          <HistoryTable key={historyKey} />
        </div>
      </main>
    </div>
  );
}
