// Behavior-parity golden guard for prompt construction in lib/generation/outline-generator.ts.
import { describe, expect, test } from 'vitest';
import { buildOutlinePrompt } from '@paatashaala/generation';

describe('buildOutlinePrompt golden output', () => {
  test('keeps ordinary lectures on slides instead of requiring interactive scenes', () => {
    const { system } = buildOutlinePrompt({ requirement: 'Explain model quantization' });
    expect(system).toContain('use these for the lecture, including worked examples');
    expect(system).toContain(
      'do not choose an interactive scene solely because a process has intermediate steps',
    );
    expect(system).not.toContain(
      'Include an interactive scene when a learner needs to watch a process advance',
    );
  });

  test('uses the selected learner level in the outline brief', () => {
    const intermediate = buildOutlinePrompt({
      requirement: 'Teach recursion',
      learnerLevel: 'intermediate',
    });
    const expert = buildOutlinePrompt({ requirement: 'Teach recursion', learnerLevel: 'expert' });
    expect(intermediate.user).toContain('## Learner Level\n\nintermediate.');
    expect(expert.user).toContain('## Learner Level\n\nexpert.');
  });

  test('pins every conditional off', () => {
    expect(
      buildOutlinePrompt(
        { requirement: 'Teach recursion to beginners' },
        { researchContext: '', teacherContext: '' },
      ),
    ).toMatchSnapshot();
  });

  test('pins every conditional on', () => {
    expect(
      buildOutlinePrompt(
        {
          requirement: '用中文讲解光合作用',
          userNickname: 'Lin',
          userBio: 'Middle-school learner',
        },
        {
          pdfText: 'Source notes about chlorophyll.',
          pdfImages: [
            {
              id: 'img_2',
              src: '',
              pageNumber: 2,
              width: 800,
              height: 600,
              description: 'Leaf cross-section',
              sourceDocumentName: 'biology.pdf',
              visionPriority: 3,
            },
          ],
          visionEnabled: true,
          imageMapping: { img_2: 'data:image/png;base64,AAAA' },
          imageGenerationEnabled: true,
          videoGenerationEnabled: true,
          researchContext: 'A current source summary.',
          teacherContext: 'Teacher Persona:\nUse a Socratic style.',
        },
      ),
    ).toMatchSnapshot();
  });

  test('pins image and media conditionals on with video off', () => {
    expect(
      buildOutlinePrompt(
        { requirement: 'Explain the water cycle with generated diagrams' },
        {
          imageGenerationEnabled: true,
          videoGenerationEnabled: false,
        },
      ),
    ).toMatchSnapshot();
  });

  test('pins source-image conditionals on with generated media off', () => {
    expect(
      buildOutlinePrompt(
        { requirement: 'Explain the labeled anatomy diagram' },
        {
          pdfImages: [
            {
              id: 'source_1',
              src: '',
              pageNumber: 4,
              width: 1200,
              height: 900,
              description: 'Labeled cross-section of a plant cell',
              sourceDocumentName: 'cell-biology.pdf',
            },
          ],
          imageGenerationEnabled: false,
          videoGenerationEnabled: false,
        },
      ),
    ).toMatchSnapshot();
  });
});
