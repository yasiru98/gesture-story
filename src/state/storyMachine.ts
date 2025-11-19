// src/state/storyMachine.ts
import { createMachine } from 'xstate';

export type StoryEvent =
  | { type: 'CHOICE_LEFT' }
  | { type: 'CHOICE_RIGHT' }
  | { type: 'CONFIRM' }
  | { type: 'BACK' }
  | { type: 'START' };

export type StoryContext = {
  // Add things later if needed
};

export const storyMachine = createMachine({
  id: 'gestureStory',

  /** XSTATE v5 typing goes inside `types` */
  types: {
    context: {} as StoryContext,
    events: {} as StoryEvent,
  },

  initial: 'intro',

  states: {
    intro: {
      on: {
        START: 'choice1',
        CHOICE_LEFT: 'choice1',
        CHOICE_RIGHT: 'choice1',
      },
    },

    choice1: {
      on: {
        CHOICE_LEFT: 'pathLeft_intro',
        CHOICE_RIGHT: 'pathRight_intro',
        BACK: 'intro',
      },
    },

    pathLeft_intro: {
      on: {
        CONFIRM: 'pathLeft_deep',
        BACK: 'choice1',
      },
    },

    pathLeft_deep: {
      on: {
        CONFIRM: 'ending_reflective',
        BACK: 'pathLeft_intro',
      },
    },

    pathRight_intro: {
      on: {
        CONFIRM: 'pathRight_deep',
        BACK: 'choice1',
      },
    },

    pathRight_deep: {
      on: {
        CONFIRM: 'ending_mysterious',
        BACK: 'pathRight_intro',
      },
    },

    ending_reflective: {
      type: 'final',
    },

    ending_mysterious: {
      type: 'final',
    },
  },
});
