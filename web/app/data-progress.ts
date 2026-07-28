export const phaseLabel = {
  adaptive_recovery: "自适应拆分恢复",
  checkpoint_reuse: "复用成功 checkpoint",
  fail_closed: "失败关闭",
} as const;

export function targetText(value: number, total: number) {
  return `${value.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")}`;
}
