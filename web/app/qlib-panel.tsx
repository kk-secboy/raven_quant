"use client";

import { ResourcePanel } from "./resource-panel";

export function QlibPanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/qlib/experiments"
      title="Qlib 实验与 Recorder"
      eyebrow="QLIB WORKFLOW / RECORDER"
      empty="尚无 Qlib 实验。"
    />
  );
}
