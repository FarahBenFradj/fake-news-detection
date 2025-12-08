import React, { useState } from "react";
import {
  Loader,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Sparkles,
  CheckCircle,
  BarChart3,
  Zap,
  Shield,
} from "lucide-react";

export default function FakeNewsDetector() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API_URL = "http://localhost:5000/predict";

  const analyzeNews = async () => {
    if (text.length < 10) {
      setError("Please enter at least 10 characters.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      if (response.ok) setResult(data);
      else setError(data.error || "Failed to analyze text.");
    } catch (err) {
      setError("Cannot connect to the server. Is Flask running?");
    } finally {
      setLoading(false);
    }
  };

  const samples = [
    { text: "MIT scientists discovered a new renewable energy source.", icon: "🔬", type: "real" },
    { text: "You won't believe what this celebrity did!", icon: "⚠️", type: "fake" },
    { text: "Government announces new budget updates.", icon: "🏛️", type: "real" },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">

      {/* HERO SECTION */}
      <div className="max-w-4xl mx-auto text-center mb-12">
        <div className="inline-flex items-center justify-center p-4 bg-white/90 backdrop-blur-xl border border-indigo-200/50 rounded-2xl shadow-lg mb-6">
          <ShieldCheck className="w-12 h-12 text-indigo-600" />
        </div>

        <h1 className="text-5xl font-extrabold bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent tracking-tight">
          Fake News Detector
        </h1>

        <p className="text-gray-600 mt-3 text-lg flex justify-center items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-500" />
          Smart AI Verification Tool
          <Sparkles className="w-5 h-5 text-indigo-500" />
        </p>

        {/* ACCURACY BADGE */}
        <div className="mt-5 inline-flex items-center gap-2 bg-white/90 backdrop-blur-sm px-5 py-2.5 rounded-full shadow-lg border border-indigo-100">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-semibold text-gray-700">98.79% Accuracy</span>
          <span className="text-gray-400">•</span>
          <span className="text-sm text-gray-600">ML Powered</span>
        </div>
      </div>

      {/* MAIN CARD */}
      <div className="max-w-4xl mx-auto bg-white/90 backdrop-blur-xl border border-gray-200 rounded-3xl shadow-2xl p-8">

        {/* STATS GRID */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-5 rounded-2xl text-center border border-indigo-200/50 hover:shadow-lg transition-shadow">
            <BarChart3 className="w-7 h-7 text-indigo-600 mx-auto mb-2" strokeWidth={2} />
            <div className="text-2xl font-bold text-indigo-800">98.79%</div>
            <div className="text-xs text-indigo-600 font-semibold mt-1">Accuracy</div>
          </div>
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-5 rounded-2xl text-center border border-blue-200/50 hover:shadow-lg transition-shadow">
            <Zap className="w-7 h-7 text-blue-600 mx-auto mb-2" strokeWidth={2} />
            <div className="text-2xl font-bold text-blue-800">Instant</div>
            <div className="text-xs text-blue-600 font-semibold mt-1">Analysis</div>
          </div>
          <div className="bg-gradient-to-br from-slate-50 to-slate-100 p-5 rounded-2xl text-center border border-slate-200/50 hover:shadow-lg transition-shadow">
            <Shield className="w-7 h-7 text-slate-600 mx-auto mb-2" strokeWidth={2} />
            <div className="text-2xl font-bold text-slate-800">Secure</div>
            <div className="text-xs text-slate-600 font-semibold mt-1">Verified</div>
          </div>
        </div>

        {/* TEXT INPUT */}
        <label className="block mb-3 text-sm font-bold text-gray-700 flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-600" /> Enter News Article or Headline
        </label>

        <div className="relative mb-5">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your news article, headline, or text here to verify its authenticity..."
            className="w-full h-48 p-5 rounded-2xl border-2 border-gray-200 bg-gradient-to-br from-white to-indigo-50/20 shadow-inner focus:ring-4 focus:ring-indigo-100 focus:border-indigo-400 outline-none font-medium text-gray-700 placeholder:text-gray-400 transition-all"
          />
          <div className="absolute bottom-4 right-4 bg-white/95 backdrop-blur-sm px-4 py-1.5 rounded-full shadow-md border border-gray-100">
            <span className={`text-sm font-bold ${text.length >= 10 ? 'text-green-600' : 'text-gray-400'}`}>
              {text.length} characters
            </span>
          </div>
        </div>

        {/* BUTTON */}
        <button
          onClick={analyzeNews}
          disabled={loading || text.length < 10}
          className={`w-full py-5 rounded-2xl font-bold text-white text-lg shadow-xl transition-all transform hover:scale-[1.01] active:scale-[0.99] ${
            text.length < 10
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-gradient-to-r from-indigo-600 to-blue-600 hover:shadow-2xl hover:from-indigo-700 hover:to-blue-700"
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-3">
              <Loader className="w-6 h-6 animate-spin" />
              Analyzing with AI...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <Sparkles className="w-5 h-5" />
              Analyze News Article
            </span>
          )}
        </button>

        {/* ERROR */}
        {error && (
          <div className="mt-6 p-5 bg-red-50 border-2 border-red-300 rounded-2xl text-red-700 flex gap-3 items-start shadow-lg">
            <AlertTriangle className="w-6 h-6 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-red-900">Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* RESULT */}
        {result && (
          <div
            className={`mt-8 p-8 rounded-3xl border-2 shadow-2xl animate-slide-up ${
              result.prediction === "REAL"
                ? "bg-gradient-to-br from-green-50 to-emerald-50 border-green-300"
                : "bg-gradient-to-br from-red-50 to-rose-50 border-red-300"
            }`}
          >
            <div className="flex items-center gap-5 mb-6 pb-6 border-b-2 border-white/50">
              <div className={`p-4 rounded-2xl shadow-xl ${
                result.prediction === "REAL"
                  ? "bg-gradient-to-br from-green-200 to-emerald-300"
                  : "bg-gradient-to-br from-red-200 to-rose-300"
              }`}>
                {result.prediction === "REAL" ? (
                  <CheckCircle className="w-12 h-12 text-green-700" strokeWidth={2.5} />
                ) : (
                  <AlertTriangle className="w-12 h-12 text-red-700" strokeWidth={2.5} />
                )}
              </div>
              <div>
                <h2 className="text-4xl font-black text-gray-800 tracking-tight mb-1">
                  {result.prediction === "REAL" ? "REAL NEWS" : "FAKE NEWS"}
                </h2>
                <p className={`text-sm font-bold ${
                  result.prediction === "REAL" ? "text-green-600" : "text-red-600"
                }`}>
                  AI Verification Complete
                </p>
              </div>
            </div>

            {/* CONFIDENCE BARS */}
            <div className="space-y-5">
              <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-md border border-green-100">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm font-bold text-gray-700 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    Real News Confidence
                  </span>
                  <span className="text-2xl font-black text-green-700">
                    {result.confidence_percentage.real}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-5 shadow-inner">
                  <div
                    className="bg-gradient-to-r from-green-500 to-emerald-500 h-5 rounded-full transition-all duration-1000 shadow-lg"
                    style={{ width: result.confidence_percentage.real }}
                  ></div>
                </div>
              </div>

              <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-md border border-red-100">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm font-bold text-gray-700 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-600" />
                    Fake News Confidence
                  </span>
                  <span className="text-2xl font-black text-red-700">
                    {result.confidence_percentage.fake}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-5 shadow-inner">
                  <div
                    className="bg-gradient-to-r from-red-500 to-rose-500 h-5 rounded-full transition-all duration-1000 shadow-lg"
                    style={{ width: result.confidence_percentage.fake }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SAMPLE SECTION */}
      <div className="max-w-4xl mx-auto mt-12">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
          <Sparkles className="w-6 h-6 text-indigo-600" />
          Try Sample Articles
        </h2>

        <div className="grid md:grid-cols-3 gap-4">
          {samples.map((s, i) => (
            <button
              key={i}
              onClick={() => setText(s.text)}
              className={`p-6 rounded-2xl border-2 shadow-md hover:shadow-xl transition-all transform hover:scale-105 active:scale-95 text-left ${
                s.type === "real"
                  ? "bg-gradient-to-br from-green-50 to-emerald-50 border-green-200 hover:border-green-300"
                  : "bg-gradient-to-br from-red-50 to-rose-50 border-red-200 hover:border-red-300"
              }`}
            >
              <div className="text-4xl mb-3">{s.icon}</div>
              <p className="text-sm text-gray-700 font-medium leading-relaxed">{s.text}</p>
            </button>
          ))}
        </div>
      </div>

      

      <style>{`
        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-slide-up {
          animation: slide-up 0.6s ease-out;
        }
      `}</style>
    </div>
  );
}