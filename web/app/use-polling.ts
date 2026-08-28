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
    let running = false;

    const schedule = () => {
      if (!stopped) timer = window.setTimeout(run, delayMs);
    };

    const run = async () => {
      if (stopped || running) return;
      timer = undefined;
      if (document.visibilityState === "hidden") {
        schedule();
        return;
      }
      running = true;
      try {
        await taskRef.current();
      } catch {
        // Panels own their visible error state. The scheduler only prevents overlap.
      } finally {
        running = false;
        schedule();
      }
    };

    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible" || running || stopped) return;
      if (timer !== undefined) window.clearTimeout(timer);
      timer = undefined;
      void run();
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    timer = window.setTimeout(run, 0);
    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", onVisibilityChange);
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [delayMs, enabled]);
}
