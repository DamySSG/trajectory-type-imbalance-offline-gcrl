# Analysis

`generate_appendix_figures.py` consumes the tracked 532-row cleaned result table
and regenerates the appendix tables and 96 response curves.

```bash
python -m analysis.generate_appendix_figures --output-root outputs/appendix
```

The `r = 1` reference is averaged within each environment-algorithm pair.
Success-rate differences are therefore descriptive percentage-point changes
relative to that pair-specific reference. They are not estimates across
independent training seeds.
