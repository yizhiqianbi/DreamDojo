import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'DreamDojo · Jokeru Evaluation Lab',
  description:
    'Interactive evaluation dashboard for the DreamDojo 2B action-conditioned world model on Jokeru robot datasets.',
  openGraph: {
    title: 'DreamDojo · Jokeru Evaluation Lab',
    description:
      'Inspect action-conditioned predictions, held-out metrics, and zero-action controls from the Jokeru post-training run.',
    images: [
      {
        url: '/dreamdojo-jokeru-og.png',
        width: 1200,
        height: 630,
        alt: 'DreamDojo Jokeru evaluation dashboard preview',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'DreamDojo · Jokeru Evaluation Lab',
    description:
      'Action-conditioned robot-video evaluation for the DreamDojo 2B Jokeru checkpoint.',
    images: ['/dreamdojo-jokeru-og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
