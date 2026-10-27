import { useEffect, useState } from "react";

/**
 * Keep a modal mounted through open/close CSS transitions.
 * - open → mount, then next frame add `visible` (enter)
 * - close → clear `visible`, unmount after `durationMs` (exit)
 */
export function useModalPresence(open: boolean, durationMs = 240) {
  const [mounted, setMounted] = useState(open);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (open) {
      setMounted(true);
      let raf2 = 0;
      const raf1 = requestAnimationFrame(() => {
        raf2 = requestAnimationFrame(() => setVisible(true));
      });
      return () => {
        cancelAnimationFrame(raf1);
        cancelAnimationFrame(raf2);
      };
    }

    setVisible(false);
    const t = window.setTimeout(() => setMounted(false), durationMs);
    return () => window.clearTimeout(t);
  }, [open, durationMs]);

  return { mounted, visible };
}
