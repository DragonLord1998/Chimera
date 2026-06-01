---
name: Expressive Neuralism
colors:
  surface: '#f8f9fa'
  surface-dim: '#d9dadb'
  surface-bright: '#f8f9fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f4f5'
  surface-container: '#edeeef'
  surface-container-high: '#e7e8e9'
  surface-container-highest: '#e1e3e4'
  on-surface: '#191c1d'
  on-surface-variant: '#414751'
  inverse-surface: '#2e3132'
  inverse-on-surface: '#f0f1f2'
  outline: '#717783'
  outline-variant: '#c1c7d3'
  surface-tint: '#0060ac'
  primary: '#005da7'
  on-primary: '#ffffff'
  primary-container: '#2976c7'
  on-primary-container: '#fdfcff'
  inverse-primary: '#a4c9ff'
  secondary: '#36656e'
  on-secondary: '#ffffff'
  secondary-container: '#baebf5'
  on-secondary-container: '#3c6b75'
  tertiary: '#8b4c11'
  on-tertiary: '#ffffff'
  tertiary-container: '#a96428'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d4e3ff'
  primary-fixed-dim: '#a4c9ff'
  on-primary-fixed: '#001c39'
  on-primary-fixed-variant: '#004883'
  secondary-fixed: '#baebf5'
  secondary-fixed-dim: '#9fced9'
  on-secondary-fixed: '#001f25'
  on-secondary-fixed-variant: '#1b4d56'
  tertiary-fixed: '#ffdcc4'
  tertiary-fixed-dim: '#ffb780'
  on-tertiary-fixed: '#2f1400'
  on-tertiary-fixed-variant: '#6f3800'
  background: '#f8f9fa'
  on-background: '#191c1d'
  surface-variant: '#e1e3e4'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 57px
    fontWeight: '700'
    lineHeight: 64px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
  title-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '500'
    lineHeight: 28px
  body-lg:
    fontFamily: Roboto Flex
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Roboto Flex
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Roboto Flex
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.1px
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  unit: 8px
  container-margin: 24px
  gutter: 16px
  section-gap: 48px
---

## Brand & Style

This design system is built on the principle of **Expressive Neuralism**. It moves away from the cold, clinical aesthetic of traditional AI tools toward a human-centric, intuitive workspace. The style is inspired by Material 3 but leans more heavily into organic transitions and soft, welcoming interfaces.

The brand personality is **intelligent yet humble**. It prioritizes cognitive ease, ensuring that complex AI training workflows feel approachable. The emotional response is one of **calm focus and creative flow**, achieved through generous white space, a muted "neural" palette, and fluid, rounded geometry. This system bridges the gap between high-performance technology and the humans who guide it.

## Colors

The palette is rooted in "Neural" tones—colors that feel cognitive and serene.

- **Primary (Neural Blue):** A soft, intelligent blue used for main actions and focus states.
- **Secondary (Muted Teal):** Represents the AI's "working" state; used for data visualization and secondary navigation.
- **Tertiary (Expressive Amber):** A carefully applied accent used only for high-priority alerts or "Aha!" moments in the AI training process.
- **Neutral (Cloud Grays):** A sophisticated range of grays with subtle blue undertones to prevent the interface from feeling "muddy."

Surfaces utilize **Tonal Offsets** rather than pure white to create a soft, non-glare environment suitable for long periods of deep work.

## Typography

The typography system pairs **Plus Jakarta Sans** for headlines with **Roboto Flex** for UI and data-heavy content. 

- **Plus Jakarta Sans** provides an "Expressive" and friendly geometric character that makes titles feel welcoming.
- **Roboto Flex** is the workhorse for the "Neural" aspect. Its variable nature ensures perfect legibility at small scales in data tables, parameter sliders, and log outputs.

Avoid uppercase styling except for very small labels. Use `title-lg` for card headers and `body-md` for the majority of the application's interactive elements to maintain a clean, uncluttered look.

## Layout & Spacing

This design system employs a **Fluid Grid** model to accommodate the dense information required by AI training platforms while maintaining a sense of openness.

- **Desktop:** 12-column grid with a maximum content width of 1440px.
- **Tablet:** 8-column grid with 24px margins.
- **Mobile:** 4-column grid with 16px margins.

Spacing follows an 8px linear scale. High-quality whitespace is a functional requirement here, not just an aesthetic choice; it helps reduce the cognitive load when users are reviewing complex neural network datasets. Group related components within cards and use the `section-gap` to clearly separate different stages of the AI workflow.

## Elevation & Depth

Visual hierarchy is established through **Tonal Layers** rather than borders. Instead of hard lines, depth is communicated by shifting the background color of a surface.

- **Level 0 (Background):** The base application color.
- **Level 1 (Surface-Container):** Slightly lighter/darker than the background; used for sidebars or secondary zones.
- **Level 2 (Active Cards):** These use **Ambient Shadows**. Shadows are extremely diffused, with a soft 12% opacity of the primary color to give the impression that the card is gently floating above the neural fabric.

Avoid using shadows on static elements. Reserve elevation for interactive components and primary content containers to guide the user's attention.

## Shapes

The shape language is **Soft and Organic**. Following the `rounded-3xl` preference, all primary containers and buttons use a high corner radius to evoke a "human-centric" feel.

- **Standard Components:** 1rem (16px) radius for buttons and inputs.
- **Cards and Modals:** 2rem (32px) radius for large surface areas.
- **Chips/Status Tags:** Fully pill-shaped (50% of height).

This softness counteracts the technical nature of AI data, making the platform feel like a cooperative tool rather than a rigid machine.

## Components

### Buttons
Primary buttons are pill-shaped with a solid color fill and no shadow. Secondary buttons use a tonal fill (a lighter version of the primary color) to maintain hierarchy without adding visual noise.

### Cards
Cards are the primary container for AI models and data sets. They should have no borders. Use a `rounded-3xl` radius and a subtle background tint. On hover, cards should slightly elevate with an ambient shadow to signal interactivity.

### Input Fields
Inputs use a "filled" style with a bottom-only indicator line in a muted tone. The container corners should be rounded at the top (12px) to match the overall organic theme.

### Chips & Progress Indicators
Use pill-shaped chips for status (e.g., "Training," "Completed"). Progress bars should be thick (8px height) with fully rounded ends, using a soft gradient of the secondary teal to show completion.

### AI Sliders
For parameter tuning, use sliders with large, "squishy" handles (24px diameter). The track should be a soft neutral-200 color, while the active fill uses the primary blue.