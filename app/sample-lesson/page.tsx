'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  Lightbulb,
  RotateCcw,
  SlidersHorizontal,
} from 'lucide-react';
import './sample-lesson.css';

const lessons = ['Start with a goal', 'Make a simple plan', 'Try a scenario'];

export default function SampleLessonPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [needs, setNeeds] = useState(50);
  const [wants, setWants] = useState(30);
  const [currency, setCurrency] = useState<'USD' | 'INR'>('USD');
  const [checked, setChecked] = useState(false);
  const savings = 100 - needs - wants;
  const isBalanced = savings >= 15 && needs >= 40;
  const amount = currency === 'USD' ? 4000 : 80000;
  const format = (share: number) =>
    new Intl.NumberFormat(currency === 'USD' ? 'en-US' : 'en-IN', {
      style: 'currency',
      currency,
      maximumFractionDigits: 0,
    }).format((amount * share) / 100);
  const changeNeeds = (value: number) => {
    setNeeds(Math.min(value, 100 - wants));
    setChecked(false);
  };
  const changeWants = (value: number) => {
    setWants(Math.min(value, 100 - needs));
    setChecked(false);
  };

  return (
    <div className="sample-app">
      <aside className="sample-rail">
        <button type="button" className="sample-brand" onClick={() => router.push('/')}>
          <img src="/paatashaala-mark.svg" alt="" aria-hidden="true" />
          <span>
            Paatashaala<small>LEARN YOUR WAY</small>
          </span>
        </button>
        <div className="sample-rail-label">SAMPLE COURSE</div>
        <h2>
          Build a budget
          <br />
          that works
        </h2>
        <p>A short preview of learning by doing.</p>
        <div className="sample-progress">
          {lessons.map((lesson, index) => (
            <button
              type="button"
              key={lesson}
              className={step === index ? 'active' : ''}
              onClick={() => setStep(index)}
            >
              <span>{index < step ? <Check size={13} /> : `0${index + 1}`}</span>
              {lesson}
            </button>
          ))}
        </div>
        <button type="button" className="sample-exit" onClick={() => router.push('/')}>
          <ArrowLeft size={16} /> Back to your space
        </button>
      </aside>
      <div className="sample-body">
        <header className="sample-header">
          <div>
            <span>
              PERSONAL FINANCE <ChevronRight size={14} /> SAMPLE LESSON
            </span>
            <h1>{lessons[step]}</h1>
          </div>
          <div className="sample-step-count">STEP {step + 1} OF 3</div>
        </header>
        <main className="sample-main">
          {step === 0 ? (
            <section className="sample-lesson-card">
              <div className="sample-overline">
                <Lightbulb size={17} /> THE IDEA
              </div>
              <h2>A good budget starts with a clear picture.</h2>
              <p>
                A budget is a plan for where your money goes. It is flexible: the right split
                depends on your income, responsibilities, and priorities.
              </p>
              <div className="sample-principle">
                <span>THE SIMPLE FRAMEWORK</span>
                <strong>Needs + Wants + Savings = 100%</strong>
                <p>Use the next step to see what changes when you adjust each part.</p>
              </div>
              <button className="sample-primary" type="button" onClick={() => setStep(1)}>
                Make a plan <ArrowRight size={17} />
              </button>
            </section>
          ) : step === 1 ? (
            <section className="sample-lesson-card">
              <div className="sample-overline">
                <SlidersHorizontal size={17} /> INTERACTIVE EXAMPLE
              </div>
              <h2>Give every dollar a job.</h2>
              <p>
                Try changing the split below. We will show what it means for a monthly income, in US
                dollars or Indian rupees.
              </p>
              <div className="sample-currency">
                <span>EXAMPLE MONTHLY INCOME</span>
                <strong>{format(100)}</strong>
                <div>
                  <button
                    type="button"
                    aria-pressed={currency === 'USD'}
                    onClick={() => setCurrency('USD')}
                  >
                    USD $
                  </button>
                  <button
                    type="button"
                    aria-pressed={currency === 'INR'}
                    onClick={() => setCurrency('INR')}
                  >
                    INR ₹
                  </button>
                </div>
              </div>
              <div className="sample-bars" aria-label="Budget allocation">
                <span style={{ width: `${needs}%` }}>Needs {needs}%</span>
                <span style={{ width: `${wants}%` }}>Wants {wants}%</span>
                <span style={{ width: `${savings}%` }}>Save {savings}%</span>
              </div>
              <div className="sample-sliders">
                <label>
                  Needs <strong>{format(needs)}</strong>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={needs}
                    onChange={(e) => changeNeeds(Number(e.target.value))}
                  />
                </label>
                <label>
                  Wants <strong>{format(wants)}</strong>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={wants}
                    onChange={(e) => changeWants(Number(e.target.value))}
                  />
                </label>
              </div>
              <div className="sample-insight">
                <Lightbulb size={18} />
                <p>
                  {savings < 15
                    ? 'You have less than 15% set aside. Could a small change create more room for savings?'
                    : `You have ${format(savings)} set aside each month. Move a slider to see how the plan changes.`}
                </p>
              </div>
              <button className="sample-primary" type="button" onClick={() => setStep(2)}>
                Try a scenario <ArrowRight size={17} />
              </button>
            </section>
          ) : (
            <section className="sample-lesson-card">
              <div className="sample-overline">
                <Check size={17} /> PUT IT TO WORK
              </div>
              <h2>Does this plan fit your goal?</h2>
              <p>
                Imagine you want to save for an emergency fund. Aim for at least 15% in savings
                while keeping at least 40% for needs. Adjust the sliders in the previous step, then
                check your plan.
              </p>
              <div className="sample-review">
                <div>
                  <span>NEEDS</span>
                  <strong>{needs}%</strong>
                </div>
                <div>
                  <span>WANTS</span>
                  <strong>{wants}%</strong>
                </div>
                <div>
                  <span>SAVINGS</span>
                  <strong>{savings}%</strong>
                </div>
              </div>
              <button type="button" className="sample-primary" onClick={() => setChecked(true)}>
                Check my plan <ArrowRight size={17} />
              </button>
              {checked && (
                <div role="status" className={`sample-feedback ${isBalanced ? 'success' : ''}`}>
                  {isBalanced
                    ? 'Nice work. This plan leaves room for essentials and your savings goal.'
                    : 'Almost there. Try moving some spending to savings while keeping enough for needs.'}
                </div>
              )}
              <button
                type="button"
                className="sample-secondary"
                onClick={() => {
                  setStep(1);
                  setChecked(false);
                }}
              >
                <RotateCcw size={15} /> Adjust the plan
              </button>
            </section>
          )}
          <aside className="sample-aside">
            <span>YOUR LEARNING COMPANION</span>
            <div className="sample-assistant-icon">
              <Lightbulb size={24} />
            </div>
            <h3>Learn, then try.</h3>
            <p>
              Real courses add more examples, practice, and a place to ask your AI teacher for help
              along the way.
            </p>
            <div className="sample-aside-line" />
            <p>
              This is a short preview. Your own course will be shaped around your topic and level.
            </p>
            <button type="button" onClick={() => router.push('/')}>
              Create your own course <ArrowRight size={16} />
            </button>
          </aside>
        </main>
      </div>
    </div>
  );
}
