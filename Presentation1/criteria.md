# Presentation criteria

The presenting group is responsible for:

- explaining the paper clearly;
- identifying its main contribution;
- explaining the methodology and key experiments;
- interpreting the results; and
- responding to critical questions.

## Connection to the lecture

Relate the Information Bottleneck objective to the lecture's distinction between a complexity term and a fitness term:

\[
\min_{p(t\mid x)} \left[I(X;T)-\beta I(T;Y)\right].
\]

- **Complexity term:** \(I(X;T)\). Lower values mean that the representation retains fewer input details.
- **Fitness term:** \(I(T;Y)\). Higher values mean that the representation retains more information useful for predicting the label.
- **Tradeoff:** \(\beta\) controls how strongly predictive fitness is valued relative to compression.

The presentation should explain the paper's claim that SGD first improves fitness by increasing \(I(T;Y)\), then reduces complexity by decreasing \(I(X;T)\).

## Final check

- Can the audience state the main contribution in one sentence?
- Are the methodology and key experiments explained clearly?
- Are the results interpreted rather than only shown?
- Are conclusions separated from hypotheses and limitations?
- Is the complexity-versus-fitness connection explicit?
- Are we prepared to discuss mutual-information estimation, binning, noise, and whether compression is necessary for generalization?
