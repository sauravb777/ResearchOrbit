import "./globals.css";
export const metadata = {
  title: "ResearchOrbit",
  description:
    "ResearchOrbit: an evidence-grounded AI research workspace for deep paper analysis, literature synthesis and claim verification.",
};
export default function RootLayout({ children }) {
  return (
    <html lang='en'>
      <body>{children}</body>
    </html>
  );
}
