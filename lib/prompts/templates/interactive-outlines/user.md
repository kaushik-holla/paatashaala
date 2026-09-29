Generate an Ultra Mode course outline based on the following requirements.

---

## User Requirements

{{requirement}}

---

## Learner Level

{{learnerLevel}}. This explicit level takes precedence over a level inferred from the wording of the request.

---

{{userProfile}}

## Language Context

Infer the course language directive by applying the decision rules from the system prompt. Key reminders:
- Requirement language = teaching language (unless overridden by explicit request or learner context)
- Foreign language learning → teach in user's native language, not the target language
- PDF language does NOT override teaching language — translate/explain document content instead

---

## Reference Materials

### PDF Content Summary

{{pdfContent}}

### Available Images

{{availableImages}}

### Web Search Results

{{researchContext}}

{{teacherContext}}

---

## Distribution Target

- Use as many explanatory slides as the learner needs to understand each mechanism before working with it.
- Use interactive scenes for guided exploration and application, not as a replacement for teaching.

## Widget Type Constraints (MANDATORY)

| Widget Type | Constraint |
|------------|-----------|
| simulation | Use when changing a variable reveals a mechanism |
| game | Use when repeated decisions and feedback deepen understanding |
| diagram | Use when tracing a structure or process helps |

## CRITICAL: Required Fields for Interactive Scenes

Every interactive scene MUST include:
- `widgetType`: One of "simulation", "diagram", "code", or "game"
- `widgetOutline`: Object with widget-specific configuration

Interactive scenes without these fields are INVALID.

## Widget Selection Guide

Choose widgets based on the content:

| Content Type | Recommended Widget |
|--------------|-------------------|
| Physics/Chemistry/Biology processes | simulation |
| Systems, processes, hierarchies | diagram |
| Programming, algorithms | code |
| Practice, challenge, application | game (action preferred) |

## Widget Design Principles (IMPORTANT)

### Simulation Widget
- Mobile-friendly: Controls MUST NOT overlap canvas
- Reset button MUST work correctly
- Touch-friendly controls (44px min)

### Diagram Widget
- First node VISIBLE on load (no blank screen)
- HIGH CONTRAST colors
- Add ICONS to nodes
- Color-code node types

### Game Widget (CRITICAL - NO BORING QUIZZES!)
- **PREFER action/puzzle games over quizzes**
- Player MUST control something (not just click answers)
- If using simulation, make it INTERACTIVE gameplay
- Example GOOD game: "Control thrust to land safely"
- Example BAD game: "Click the correct answer"
- `gameType` should be "action", "puzzle", or "strategy", NOT just "quiz"

### Example: Good vs Bad Game Outline

❌ **BAD (boring quiz):**
```json
{
  "widgetType": "game",
  "widgetOutline": {
    "gameType": "quiz",
    "questionCount": 5
  }
}
```

✅ **GOOD (interactive game):**
```json
{
  "widgetType": "game",
  "widgetOutline": {
    "gameType": "action",
    "challenge": "控制推力使飞船安全着陆",
    "playerControls": ["thrust_slider"]
  }
}
```

**Final reminder**: your entire response must be a JSON **object** with exactly three top-level keys — `languageDirective` (string, inferred via the Language Inference rules in the system prompt), `courseTitle` (string, ≤30 chars, in the teaching language), and `outlines` (array of scene objects). Do not return a bare array. Do not wrap in prose or code fences.
