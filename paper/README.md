# JOSS manuscript

The submission to the [Journal of Open Source Software](https://joss.theoj.org/). `.github/workflows/draft-pdf.yml` builds a draft PDF from `paper.md` on every push that touches this directory, and uploads it as a workflow artifact.

| File | Authored or generated | Notes |
| --- | --- | --- |
| `paper.md` | authored | The manuscript. Every number in it is recomputable — see below. |
| `paper.bib` | authored | Bibliography. |
| `software_architecture.mmd` | authored | Mermaid source for the architecture figure. The render command is in a comment on its second line. |
| `software_architecture.png` | generated | Rendered from the `.mmd` with `mmdc`. Regenerate it rather than editing it. |
| `benchmark_summary.png` | generated | Copied here by `benchmarks/make_figures.py`, which is the only thing that should write it. Editing it by hand is silently undone the next time the figures are drawn. |

## Regenerating the figures

The architecture diagram needs the Mermaid CLI:

```bash
cd paper
mmdc -i software_architecture.mmd -o software_architecture.png -b white -w 1500 -s 2
```

The benchmark figure is drawn from the measured result tables and needs no GPU, no images and no solver:

```bash
cd benchmarks
pixi run python make_figures.py
```

That publishes `benchmark_summary.png` here and the full figure set into `docs/src/assets/`. Drawing the figures from a tree whose `benchmarks/results/` is unchanged reproduces the committed PNGs byte for byte, which is what makes a stale figure detectable rather than merely unlikely.

## Checking the numbers

`paper.md` states its results as prose rather than generating them, so they are verified by recomputation instead:

```bash
cd benchmarks
pixi run python paper_numbers.py    # every headline number in paper.md
pixi run python docs_numbers.py     # the tables and cited cells in docs/src/benchmark.md
```

Both read `benchmarks/results/` and nothing else. Run them before and after any change to the dataset and diff the output: an empty diff means the prose is still true.
