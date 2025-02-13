'use client';

import React, { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [title, setTitle] = useState('Data Scientist');
  const [company, setCompany] = useState('H&M');
  const [location, setLocation] = useState('Stockholm');
  const [description, setDescription] = useState('We are looking for a Data Scientist');
  const [skills, setSkills] = useState('python,sql,ml');
  const [experienceFrom, setExperienceFrom] = useState(0);
  const [experienceTo, setExperienceTo] = useState(3);
  const [predictedSalary, setPredictedSalary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    try {
      setError(null);
      const response = await fetch(process.env.NEXT_PUBLIC_API_ENDPOINT || 'http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title,
          company,
          location,
          description,
          skills,
          experience_from: experienceFrom,
          experience_to: experienceTo,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setPredictedSalary(data.predicted_salary);
    } catch (e: any) {
      setError(e.message);
      console.error("Prediction error:", e);
    }
  };

  return (
    <main className={styles.main}>
      <h1>Salary Prediction</h1>
      {error && <div className="error">Error: {error}</div>}
      <div>
        <label htmlFor="title">Job Title:</label>
        <input type="text" id="title" value={title} onChange={(e) => setTitle(e.target.value)} />
      </div>
      <div>
        <label htmlFor="company">Company:</label>
        <input type="text" id="company" value={company} onChange={(e) => setCompany(e.target.value)} />
      </div>
      <div>
        <label htmlFor="location">Location:</label>
        <input type="text" id="location" value={location} onChange={(e) => setLocation(e.target.value)} />
      </div>
      <div>
        <label htmlFor="description">Job Description:</label>
        <textarea id="description" value={description} onChange={(e) => setDescription(e.target.value)} />
      </div>
      <div>
        <label htmlFor="skills">Required Skills (comma-separated):</label>
        <textarea id="skills" value={skills} onChange={(e) => setSkills(e.target.value)} />
      </div>
      <div>
        <label htmlFor="experienceFrom">Minimum Years of Experience:</label>
        <input
          type="number"
          id="experienceFrom"
          value={experienceFrom}
          onChange={(e) => setExperienceFrom(parseInt(e.target.value))}
        />
      </div>
      <div>
        <label htmlFor="experienceTo">Maximum Years of Experience:</label>
        <input
          type="number"
          id="experienceTo"
          value={experienceTo}
          onChange={(e) => setExperienceTo(parseInt(e.target.value))}
        />
      </div>

      <button onClick={handleSubmit}>Predict Salary</button>

      {predictedSalary && (
        <div className="result">
          Predicted Salary: SEK {predictedSalary}
        </div>
      )}
    </main>
  );
}
