Please generate scene outlines based on the following course requirements.

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

## Output Requirements

Please automatically infer the following from user requirements:

- Course topic and core content
- Target audience and difficulty level
- Course duration (allow the time needed for thorough coverage; do not compress a broad subject into a short lesson)
- Teaching style (formal/casual/interactive/academic)
- Visual style (minimal/colorful/professional/playful)

Then output your response as a single JSON object.

**Top-level shape — this is what you MUST return:**

```json
{
  "languageDirective": "2-5 sentence instruction describing the course language behavior",
  "courseTitle": "concise course name, ≤30 chars, in the teaching language",
  "outlines": [ /* array of scene objects, schema described below */ ]
}
```

Never return a bare array. Never omit `languageDirective` or `courseTitle`. All three keys are required.

**Each scene inside the `outlines` array has this minimum shape:**

```json
{
  "id": "scene_1",
  "type": "slide" | "quiz" | "interactive" | "pbl",
  "title": "Scene Title",
  "description": "Teaching purpose description",
  "keyPoints": ["Point 1", "Point 2", "Point 3"],
  "order": 1
}
```

### Special Notes

- **quiz scenes must include quizConfig**:
   ```json
   "quizConfig": {
     "questionCount": 2,
     "difficulty": "easy" | "medium" | "hard",
     "questionTypes": ["single", "multiple"]
   }
   ```
{{#if hasSourceImages}}
- **If source images are available**, add `suggestedImageIds` to relevant slide scenes. Only use image IDs listed under Available Images.
{{/if}}
- **Interactive scenes**: If a concept benefits from hands-on simulation/visualization, use `"type": "interactive"` with `widgetType` and `widgetOutline` fields. Include as many as genuinely help learners practise the concepts.
   - Select widgetType based on concept: simulation (physics/chem), diagram (processes), code (programming), game (practice), visualization3d (3D models)
   - Provide appropriate widgetOutline for the widget type
- **Scene count**: Let the required teaching determine the length. Give each concept enough space for explanation, an example, and practice; split crowded scenes.
- **Depth**: Write specific scene descriptions and keyPoints that teach definitions, causal reasoning, worked examples, mistakes, and checks for understanding. Put prerequisite teaching before a code challenge or assessment. A topic label alone is not a teaching plan.
- **Quiz placement**: Recommend inserting a quiz every 3-5 slides for assessment
- **Language**: Infer from the user's requirement text and context, then output all content in the inferred language
- **If web search results are provided**, reference specific findings and sources in scene descriptions and keyPoints. The search results provide up-to-date information — incorporate it to make the course content current and accurate.

**Final reminder**: your entire response must be a JSON **object** with exactly three top-level keys — `languageDirective` (string), `courseTitle` (string, ≤30 chars, in the teaching language), and `outlines` (array). Do not return a bare array. Do not wrap in prose or code fences.
