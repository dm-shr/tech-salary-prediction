import { NextResponse } from "next/server"

export async function POST(request: Request) {
  const body = await request.json()

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      throw new Error("Failed to predict salary")
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error predicting salary:", error)
    return NextResponse.json({ error: "Failed to predict salary" }, { status: 500 })
  }
}
