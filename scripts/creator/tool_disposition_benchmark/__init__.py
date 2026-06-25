"""Tool-disposition benchmark (CREATOR-derived).

Measures tool *disposition*: given a scarce script-writing budget (0.1N) and many
DISTINCT hard-arithmetic problems answered one at a time in a single persistent session,
does a model build a small kit of general, composable primitives it reuses across problems
— or burn the budget on one-offs (then fail the precision-grind) or grind by hand (and miss
the required significant figures)?

See docs / the plan for the four metrics: solve rate, efficiency, reusability, recognition.
"""
