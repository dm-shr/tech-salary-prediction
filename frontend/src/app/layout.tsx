import { Inter } from "next/font/google"
import "./globals.css"
import { metadata } from "./metadata"

const inter = Inter({ subsets: ["latin"] })

export { metadata }

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-100">
          <main className="container mx-auto px-4 py-0"> {/* Reduced py-8 to py-4 */}
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
