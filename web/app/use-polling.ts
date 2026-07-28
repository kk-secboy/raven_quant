"use client";

import { useEffect, useRef } from "react";

export function usePolling(task: () => void | Promise<void>, delayMs: number, enabled = true) {
  const taskRef = useRef(task);

  useEffect(() => {
    taskRef.current = task;
  }, [task]);

  useEffect(() => {
    if (!enabled) return;

    let stopped = false;
    let timer: number | undefined;

    const run = async () => {
      try {
        await taskRef.current();
      } catch {
        // Panels own their visible error state. The scheduler only prevents overlap.
      } finally {
        if (!stopped) timer = window.setTimeout(run, delayMs);
      }
    };

    timer = window.setTimeout(run, 0);
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [delayMs, enabled]);
}
