import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuantLab · 量化研究系统",
  description: "基于 Tushare、Qlib 与 RD-Agent 的本地量化研究平台。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
