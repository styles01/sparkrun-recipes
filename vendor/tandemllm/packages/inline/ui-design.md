# UI design

Build interfaces that look designed, not assembled.

1. Start from the content. Decide what the user must see first on the page, then second. One primary action per screen; make it the strongest element.
2. Use a spacing scale (4, 8, 12, 16, 24, 32, 48 px) and nothing between. Align everything to it. Group related items with less space, separate groups with more.
3. Type: one family for UI text, at most one for display. A scale such as 12, 14, 16, 20, 24, 32 px. Body text 14 to 16 px, line height 1.4 to 1.6, lines at most 75 characters.
4. Colour: define tokens first (background, surface, border, text, muted text, one accent, success, warning, danger). Text contrast at least 4.5:1. Use the accent for the primary action and the current state only.
5. Layout: CSS grid or flexbox, no absolute positioning for layout. Mobile first: one column at 390 px with no horizontal scroll, then widen at breakpoints. Tap targets at least 44 px.
6. States: every component needs hover, focus-visible, active, disabled, loading, empty and error states. Never leave a blank area while data loads.
7. Motion: 150 to 250 ms, ease-out, only to explain a change. Respect prefers-reduced-motion.
8. Accessibility: semantic HTML first, labels for every input, visible focus, keyboard reachable controls, alt text.
9. Check your work: open the page at 390 px and at 1440 px, compare the result with the brief, and fix alignment and spacing before adding features.
