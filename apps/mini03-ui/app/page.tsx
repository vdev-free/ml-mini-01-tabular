"use client";

import { useState } from "react";

type ApiResponse = {
  segment: string;
  cluster: number;
};

export default function Page() {
  const [purchases, setPurchases] = useState<string>("18");
  const [spend, setSpend] = useState<string>("1300");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const purchasesNum = Number(purchases);
    const spendNum = Number(spend);

    if (!Number.isFinite(purchasesNum) || purchasesNum < 0) {
      setLoading(false);
      setError("purchases_30d має бути числом ≥ 0");
      return;
    }
    if (!Number.isFinite(spendNum) || spendNum < 0) {
      setLoading(false);
      setError("spend_30d має бути числом ≥ 0");
      return;
    }

    try {
      const r = await fetch("http://127.0.0.1:8003/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          purchases_30d: purchasesNum,
          spend_30d: spendNum,
        }),
      });

      if (!r.ok) {
        const text = await r.text();
        throw new Error(`API error ${r.status}: ${text}`);
      }

      const data = (await r.json()) as ApiResponse;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-6">
      <div className="mx-auto max-w-xl space-y-6">
        <h1 className="text-2xl font-semibold">
          Mini03 — Customer Segmentation Demo
        </h1>

        <form onSubmit={onSubmit} className="rounded-xl border p-4 space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Purchases (30d)</label>
            <input
              className="w-full rounded-md border px-3 py-2"
              inputMode="numeric"
              value={purchases}
              onChange={(e) => setPurchases(e.target.value)}
              placeholder="наприклад 9"
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">Spend (30d)</label>
            <input
              className="w-full rounded-md border px-3 py-2"
              inputMode="numeric"
              value={spend}
              onChange={(e) => setSpend(e.target.value)}
              placeholder="наприклад 350"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="rounded-md border px-4 py-2 disabled:opacity-60"
          >
            {loading ? "Segmenting..." : "Segment"}
          </button>
        </form>

        {error && (
          <div className="rounded-xl border p-4">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {result && (
          <div className="rounded-xl border p-4">
            <p className="font-medium">Result</p>
            <p className="text-sm">
              Segment: <b>{result.segment}</b>
            </p>
            <p className="text-sm">Cluster: {result.cluster}</p>
          </div>
        )}

        <p className="text-xs opacity-70">
          API: http://127.0.0.1:8003 (запусти FastAPI паралельно)
        </p>
      </div>
    </main>
  );
}
