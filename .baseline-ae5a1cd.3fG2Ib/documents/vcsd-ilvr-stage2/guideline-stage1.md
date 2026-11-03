# Stage 1 evaluation guide

Stage 1 has an independently evaluable, paper-defined continuous-latent path.
After `<|latent_start|>`, evaluation feeds the previous hidden state back for
exactly the configured latent length before continuing from `<|latent_end|>`.
Static `<|latent_pad|>` generation is not the Stage 1 evaluation method; it is
only a diagnostic control for measuring the effect of continuous feedback.

Stage 1 full and PEFT artifacts use their own validated loaders. They remain
separate from Stage 2 full and PEFT loading and Stage 2's strict resolver; an
evaluation must preserve the artifact's Stage 1 tag rather than relabel it.

Answer scoring reports both denominators: coverage accuracy is `correct / 400`,
and valid-label accuracy is `correct / 398` when the two known invalid labels
are excluded. Missing, blank, or otherwise invalid gold labels never match a
prediction, including an empty prediction.
